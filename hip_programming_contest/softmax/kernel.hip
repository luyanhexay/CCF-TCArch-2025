#include "main.h"

// 实现数值稳定 softmax：
// 1) 归约求全局最大值 m
// 2) 基于 m 计算 exp(x_i - m) 的和 S 的归约
// 3) 第二遍计算输出 y_i = exp(x_i - m) / max(S, 1e-12)

#include <cstdio>
#include <cstdlib>
#include <math.h>

#ifndef TPB
#define TPB 256
#endif

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        std::exit(1); \
    } \
} while(0)

// 组合两段 softmax 运行统计 (m, l) 的合并运算
// 输入 a=(m_a, l_a), b=(m_b, l_b)
// 输出 (m, l):
//   m = max(m_a, m_b)
//   l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
__device__ inline void combineRunningStats(float& m_acc, double& l_acc,
                                           float m_other, double l_other) {
    float m_new = m_acc > m_other ? m_acc : m_other;
    double l_new = 0.0;
    if (l_acc != 0.0) {
        l_new += l_acc * exp((double)(m_acc - m_new));
    }
    if (l_other != 0.0) {
        l_new += l_other * exp((double)(m_other - m_new));
    }
    m_acc = m_new;
    l_acc = l_new;
}

__global__ void reduceMaxKernel(const float* __restrict__ x, float* __restrict__ blockMax, int n) {
    extern __shared__ float sdataF[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float localMax = -FLT_MAX;
    // 向量化读取 float4
    const int numVec = n >> 2; // n/4
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int idx4 = blockIdx.x * blockDim.x + tid;
    int stride4 = stride;
    for (; idx4 < numVec; idx4 += stride4) {
        float4 v4 = x4[idx4];
        localMax = v4.x > localMax ? v4.x : localMax;
        localMax = v4.y > localMax ? v4.y : localMax;
        localMax = v4.z > localMax ? v4.z : localMax;
        localMax = v4.w > localMax ? v4.w : localMax;
    }
    // 处理尾部不足 4 个元素
    int base = numVec << 2;
    int idxTail = base + blockIdx.x * blockDim.x + tid;
    for (; idxTail < n; idxTail += stride) {
        float v = x[idxTail];
        localMax = v > localMax ? v : localMax;
    }

    // Wavefront 内归约 (shuffle)
    float warpVal = localMax;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_down(warpVal, offset);
        warpVal = warpVal > other ? warpVal : other;
    }
    int lane = tid & (warpSize - 1);
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) {
        sdataF[warpId] = warpVal;
    }
    __syncthreads();

    if (warpId == 0) {
        float blockVal = -FLT_MAX;
        if (lane < numWarps) {
            blockVal = sdataF[lane];
        }
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float other = __shfl_down(blockVal, offset);
            blockVal = blockVal > other ? blockVal : other;
        }
        if (lane == 0) {
            blockMax[blockIdx.x] = blockVal;
        }
    }
}

__global__ void reduceSumExpKernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   double* __restrict__ blockSums,
                                   int n,
                                   const float* __restrict__ mPtr) {
    extern __shared__ double sdataD[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float m = *mPtr;
    double localSum = 0.0;
    // 向量化读取/写入 float4
    const int numVec = n >> 2; // n/4
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int idx4 = blockIdx.x * blockDim.x + tid;
    int stride4 = stride;
    for (; idx4 < numVec; idx4 += stride4) {
        float4 v4 = x4[idx4];
        float s0 = v4.x - m;
        float s1 = v4.y - m;
        float s2 = v4.z - m;
        float s3 = v4.w - m;
        double t0 = (double)expf(s0);
        double t1 = (double)expf(s1);
        double t2 = (double)expf(s2);
        double t3 = (double)expf(s3);
        float4 out4;
        out4.x = (float)t0;
        out4.y = (float)t1;
        out4.z = (float)t2;
        out4.w = (float)t3;
        y4[idx4] = out4; // 缓存 t_i
        localSum += (t0 + t1 + t2 + t3);
    }
    // 尾部处理
    int base = numVec << 2;
    int idxTail = base + idx;
    for (; idxTail < n; idxTail += stride) {
        float shifted = x[idxTail] - m;
        double t = (double)expf(shifted);
        y[idxTail] = (float)t;
        localSum += t;
    }

    // Wavefront 内归约 (shuffle)
    double warpSum = localSum;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        double other = __shfl_down(warpSum, offset);
        warpSum += other;
    }
    int lane = tid & (warpSize - 1);
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tid / warpSize;

    if (lane == 0) {
        sdataD[warpId] = warpSum;
    }
    __syncthreads();

    if (warpId == 0) {
        double blockVal = 0.0;
        if (lane < numWarps) {
            blockVal = sdataD[lane];
        }
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            double other = __shfl_down(blockVal, offset);
            blockVal += other;
        }
        if (lane == 0) {
            blockSums[blockIdx.x] = blockVal;
        }
    }
}

__global__ void softmaxNormalizeKernel(float* __restrict__ y, int n, const double* __restrict__ sumExpPtr) {
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sumExp = *sumExpPtr;
    float denom = (float)fmax(1e-12, sumExp);
    // 向量化归一化
    const int numVec = n >> 2; // n/4
    float4* y4 = reinterpret_cast<float4*>(y);
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride4 = stride;
    for (; idx4 < numVec; idx4 += stride4) {
        float4 v4 = y4[idx4];
        v4.x = v4.x / denom;
        v4.y = v4.y / denom;
        v4.z = v4.z / denom;
        v4.w = v4.w / denom;
        y4[idx4] = v4;
    }
    // 尾部处理
    int base = numVec << 2;
    int idxTail = base + idx;
    for (; idxTail < n; idxTail += stride) {
        y[idxTail] = y[idxTail] / denom;
    }
}

// 设备端最终归约：将长度为 len 的数组归约为 1 个标量
__global__ void reduceFinalMaxKernel(const float* __restrict__ inVals, float* __restrict__ outVal, int len) {
    extern __shared__ float sdataF[];
    const int tid = threadIdx.x;

    float localMax = -FLT_MAX;
    for (int i = tid; i < len; i += blockDim.x) {
        float v = inVals[i];
        localMax = v > localMax ? v : localMax;
    }

    // Wavefront 内归约
    float warpVal = localMax;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_down(warpVal, offset);
        warpVal = warpVal > other ? warpVal : other;
    }
    int lane = tid & (warpSize - 1);
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) {
        sdataF[warpId] = warpVal;
    }
    __syncthreads();
    if (warpId == 0) {
        float blockVal = -FLT_MAX;
        if (lane < numWarps) blockVal = sdataF[lane];
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float other = __shfl_down(blockVal, offset);
            blockVal = blockVal > other ? blockVal : other;
        }
        if (lane == 0) {
            *outVal = blockVal;
        }
    }
}

__global__ void reduceFinalSumKernel(const double* __restrict__ inVals, double* __restrict__ outVal, int len) {
    extern __shared__ double sdataD[];
    const int tid = threadIdx.x;

    double localSum = 0.0;
    for (int i = tid; i < len; i += blockDim.x) {
        localSum += inVals[i];
    }

    // Wavefront 内归约
    double warpVal = localSum;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        double other = __shfl_down(warpVal, offset);
        warpVal += other;
    }
    int lane = tid & (warpSize - 1);
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    int warpId = tid / warpSize;
    if (lane == 0) {
        sdataD[warpId] = warpVal;
    }
    __syncthreads();
    if (warpId == 0) {
        double blockVal = 0.0;
        if (lane < numWarps) blockVal = sdataD[lane];
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            double other = __shfl_down(blockVal, offset);
            blockVal += other;
        }
        if (lane == 0) {
            *outVal = blockVal;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    // 选择合理的 grid 配置
    int threads = TPB;
    // 将 blocks 控制在上限以内（网格步长循环可以覆盖全体）
    int maxBlocks = 65535;
    int blocks = (N + threads - 1) / threads;
    if (blocks > maxBlocks) blocks = maxBlocks;

    // 设备内存分配
    float* d_x = nullptr;
    float* d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, sizeof(float) * (size_t)N));
    HIP_CHECK(hipMalloc(&d_y, sizeof(float) * (size_t)N));

    // 注册页锁定内存以提升传输带宽
    HIP_CHECK(hipHostRegister((void*)input, sizeof(float) * (size_t)N, 0));
    HIP_CHECK(hipHostRegister((void*)output, sizeof(float) * (size_t)N, 0));

    HIP_CHECK(hipMemcpy(d_x, input, sizeof(float) * (size_t)N, hipMemcpyHostToDevice));

    // O3: 融合到“两趟/两读一写”方案：
    // Pass 1: 以在线法求 (m,l)，仅从 d_x 读取一次，不写中间 y
    // Pass 2: 使用 (m,l) 直接从 d_x 计算输出写入 d_y（写一次）

    // 为 Pass1 分配块级 (m,l) 缓冲
    float* d_blockM = nullptr;
    double* d_blockL = nullptr;
    HIP_CHECK(hipMalloc(&d_blockM, sizeof(float) * (size_t)blocks));
    HIP_CHECK(hipMalloc(&d_blockL, sizeof(double) * (size_t)blocks));

    // 计算动态共享内存：每个 warp 存 1 个 (m,l)
    int numWarpsHost = (threads + warpSize - 1) / warpSize;
    size_t smemOnline = (sizeof(double) + sizeof(float)) * (size_t)numWarpsHost;

    // 声明/定义内核（见下方新内核定义）
    extern __global__ void onlinePartialKernel(const float* __restrict__, float* __restrict__, double* __restrict__, int);
    extern __global__ void onlineFinalReduceKernel(const float* __restrict__, const double* __restrict__, float* __restrict__, double* __restrict__, int);
    extern __global__ void softmaxWriteKernel(const float* __restrict__, float* __restrict__, int, const float* __restrict__, const double* __restrict__);

    // Pass1: 各块生成 (m_b, l_b)
    onlinePartialKernel<<<blocks, threads, smemOnline>>>(d_x, d_blockM, d_blockL, N);
    HIP_CHECK(hipGetLastError());

    // 设备端最终归约得到 (m, l)
    float* d_m = nullptr;
    double* d_l = nullptr;
    HIP_CHECK(hipMalloc(&d_m, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_l, sizeof(double)));
    onlineFinalReduceKernel<<<1, threads, smemOnline>>>(d_blockM, d_blockL, d_m, d_l, blocks);
    HIP_CHECK(hipGetLastError());

    // Pass2: 直接写出 y = exp(x - m) / max(l, 1e-12)
    softmaxWriteKernel<<<blocks, threads>>>(d_x, d_y, N, d_m, d_l);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(output, d_y, sizeof(float) * (size_t)N, hipMemcpyDeviceToHost));

    // 清理资源
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_blockM));
    HIP_CHECK(hipFree(d_blockL));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_l));
    HIP_CHECK(hipHostUnregister((void*)input));
    HIP_CHECK(hipHostUnregister((void*)output));
}

// --- 新增：基于在线法的 (m,l) 分块计算 ---
__global__ void onlinePartialKernel(const float* __restrict__ x,
                                    float* __restrict__ blockM,
                                    double* __restrict__ blockL,
                                    int n) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float m_local = -INFINITY;
    double l_local = 0.0;

    // 向量化遍历
    int numVec = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int idx4 = blockIdx.x * blockDim.x + tid;
    int stride4 = stride;
    for (; idx4 < numVec; idx4 += stride4) {
        float4 v4 = x4[idx4];
        // 依次合并四个元素
        combineRunningStats(m_local, l_local, v4.x, 1.0);
        combineRunningStats(m_local, l_local, v4.y, 1.0);
        combineRunningStats(m_local, l_local, v4.z, 1.0);
        combineRunningStats(m_local, l_local, v4.w, 1.0);
    }
    // 尾部
    int base = numVec << 2;
    int idx = base + blockIdx.x * blockDim.x + tid;
    for (; idx < n; idx += stride) {
        float vx = x[idx];
        combineRunningStats(m_local, l_local, vx, 1.0);
    }

    // 以两级归约方式合并到块级 (m,l)
    extern __shared__ unsigned char smemRaw[];
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    double* sL = reinterpret_cast<double*>(smemRaw);
    float* sM = reinterpret_cast<float*>(sL + numWarps);

    // Warp 内归约
    float m_w = m_local;
    double l_w = l_local;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float m_o = __shfl_down(m_w, offset);
        double l_o = __shfl_down(l_w, offset);
        combineRunningStats(m_w, l_w, m_o, l_o);
    }
    int lane = tid & (warpSize - 1);
    int warpId = tid / warpSize;
    if (lane == 0) {
        sM[warpId] = m_w;
        sL[warpId] = l_w;
    }
    __syncthreads();

    // Warp 间归约（由第一个 warp 完成）
    if (warpId == 0) {
        float m_acc = -INFINITY;
        double l_acc = 0.0;
        if (lane < numWarps) {
            m_acc = sM[lane];
            l_acc = sL[lane];
        }
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float m_o = __shfl_down(m_acc, offset);
            double l_o = __shfl_down(l_acc, offset);
            combineRunningStats(m_acc, l_acc, m_o, l_o);
        }
        if (lane == 0) {
            blockM[blockIdx.x] = m_acc;
            blockL[blockIdx.x] = l_acc;
        }
    }
}

// --- 新增：最终归约得到 (m, l) ---
__global__ void onlineFinalReduceKernel(const float* __restrict__ inM,
                                        const double* __restrict__ inL,
                                        float* __restrict__ outM,
                                        double* __restrict__ outL,
                                        int len) {
    const int tid = threadIdx.x;

    // 线程私有累积
    float m_local = -INFINITY;
    double l_local = 0.0;
    for (int i = tid; i < len; i += blockDim.x) {
        float m_i = inM[i];
        double l_i = inL[i];
        combineRunningStats(m_local, l_local, m_i, l_i);
    }

    // 两级归约
    extern __shared__ unsigned char smemRaw[];
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    double* sL = reinterpret_cast<double*>(smemRaw);
    float* sM = reinterpret_cast<float*>(sL + numWarps);

    float m_w = m_local;
    double l_w = l_local;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float m_o = __shfl_down(m_w, offset);
        double l_o = __shfl_down(l_w, offset);
        combineRunningStats(m_w, l_w, m_o, l_o);
    }
    int lane = tid & (warpSize - 1);
    int warpId = tid / warpSize;
    if (lane == 0) {
        sM[warpId] = m_w;
        sL[warpId] = l_w;
    }
    __syncthreads();
    if (warpId == 0) {
        float m_acc = -INFINITY;
        double l_acc = 0.0;
        if (lane < numWarps) {
            m_acc = sM[lane];
            l_acc = sL[lane];
        }
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float m_o = __shfl_down(m_acc, offset);
            double l_o = __shfl_down(l_acc, offset);
            combineRunningStats(m_acc, l_acc, m_o, l_o);
        }
        if (lane == 0) {
            *outM = m_acc;
            *outL = l_acc;
        }
    }
}

// --- 新增：直接写出 y ---
__global__ void softmaxWriteKernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int n,
                                   const float* __restrict__ mPtr,
                                   const double* __restrict__ lPtr) {
    const int stride = blockDim.x * gridDim.x;
    float m = *mPtr;
    double l = *lPtr;
    float denom = (float)fmax(1e-12, l);

    // 向量化归一化写出
    int numVec = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride4 = stride;
    for (; idx4 < numVec; idx4 += stride4) {
        float4 v4 = x4[idx4];
        v4.x = expf(v4.x - m) / denom;
        v4.y = expf(v4.y - m) / denom;
        v4.z = expf(v4.z - m) / denom;
        v4.w = expf(v4.w - m) / denom;
        y4[idx4] = v4;
    }
    int base = numVec << 2;
    int idx = base + blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += stride) {
        y[idx] = expf(x[idx] - m) / denom;
    }
}