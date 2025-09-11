#include "main.h"

// 实现数值稳定 softmax：
// 使用在线统计算法，在一次遍历中同时计算最大值和求和
// 然后直接写出最终结果

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdint.h>
#include <cstring> // For memcpy

// 最优参数配置（基于性能测试结果）
#ifndef TPB
#define TPB 256
#endif

#ifndef GRID_MULTIPLIER
#define GRID_MULTIPLIER 4
#endif

#ifndef PINNED_MEMORY_THRESHOLD
#define PINNED_MEMORY_THRESHOLD 100000
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

// 安全的double类型shuffle函数
// 将double拆分为两个32位整数进行shuffle，然后重新组合
__device__ inline double shfl_down_double(double val, int offset) {
    // 将double转换为两个32位整数
    union {
        double d;
        int2 i;
    } converter;
    converter.d = val;
    
    // 分别shuffle两个部分
    int part0 = __shfl_down(converter.i.x, offset);
    int part1 = __shfl_down(converter.i.y, offset);
    
    // 重新组合为double
    converter.i.x = part0;
    converter.i.y = part1;
    return converter.d;
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    size_t dataSize = sizeof(float) * (size_t)N;

    // --- 优化点: 使用 hipHostMalloc 分配页锁定内存 ---
    // 这比在普通内存上调用 hipHostRegister 快得多
    float* h_pinned_input = nullptr;
    float* h_pinned_output = nullptr;
    HIP_CHECK(hipHostMalloc(&h_pinned_input, dataSize));
    HIP_CHECK(hipHostMalloc(&h_pinned_output, dataSize));
    
    // 将用户数据拷贝到我们准备好的页锁定内存中
    // 这是一个快速的 host-to-host 拷贝
    memcpy(h_pinned_input, input, dataSize);

    // 获取GPU硬件信息以优化网格大小
    int numCU;
    HIP_CHECK(hipDeviceGetAttribute(&numCU, hipDeviceAttributeMultiprocessorCount, 0));
    
    // 选择合理的线程配置
    int threads = TPB;
    
    // 启动足够多的线程块来饱和GPU，每个CU启动GRID_MULTIPLIER个块
    int blocks = numCU * GRID_MULTIPLIER;
    
    // 如果N很小，不需要启动这么多块
    int required_blocks = (N + threads - 1) / threads;
    if (blocks > required_blocks) {
        blocks = required_blocks;
    }
    
    // 限制最大块数
    int maxBlocks = 65535;
    if (blocks > maxBlocks) blocks = maxBlocks;

    // 使用默认流，完全避免流创建开销
    hipStream_t stream = hipStreamDefault;

    // 设备内存分配
    float* d_x = nullptr;
    float* d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, dataSize));
    HIP_CHECK(hipMalloc(&d_y, dataSize));

    // 异步H2D传输 - 从页锁定内存开始，以获得最佳性能
    HIP_CHECK(hipMemcpyAsync(d_x, h_pinned_input, dataSize, hipMemcpyHostToDevice, stream));

    // 计算动态共享内存：每个 warp 存 1 个 (m,l)
    int numWarpsHost = (threads + warpSize - 1) / warpSize;
    size_t smemOnline = (sizeof(double) + sizeof(float)) * (size_t)numWarpsHost + 8; // 额外字节保证8字节对齐

    // 计算共享内存大小：用于缓存输入数据和归约
    size_t smemDataSize = sizeof(float) * (size_t)threads; // 每个线程块缓存的数据
    size_t totalSmemSize = smemOnline + smemDataSize;

    // 为块级 (m,l) 缓冲分配内存
    float* d_blockM = nullptr;
    double* d_blockL = nullptr;
    HIP_CHECK(hipMalloc(&d_blockM, sizeof(float) * (size_t)blocks));
    HIP_CHECK(hipMalloc(&d_blockL, sizeof(double) * (size_t)blocks));

    // 声明内核函数
    extern __global__ void onlinePartialKernel(const float* __restrict__, float* __restrict__, double* __restrict__, int);
    extern __global__ void onlineFinalReduceKernel(const float* __restrict__, const double* __restrict__, float* __restrict__, double* __restrict__, int);
    extern __global__ void softmaxWriteKernel(const float* __restrict__, float* __restrict__, int, const float* __restrict__, const double* __restrict__);

    // Pass1: 各块生成 (m_b, l_b) - 异步执行
    onlinePartialKernel<<<blocks, threads, smemOnline, stream>>>(d_x, d_blockM, d_blockL, N);
    HIP_CHECK(hipGetLastError());

    // 设备端最终归约得到 (m, l)
    float* d_m = nullptr;
    double* d_l = nullptr;
    HIP_CHECK(hipMalloc(&d_m, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_l, sizeof(double)));
    onlineFinalReduceKernel<<<1, threads, smemOnline, stream>>>(d_blockM, d_blockL, d_m, d_l, blocks);
    HIP_CHECK(hipGetLastError());

    // Pass2: 直接写出 y = exp(x - m) / max(l, 1e-12) - 异步执行
    softmaxWriteKernel<<<blocks, threads, 0, stream>>>(d_x, d_y, N, d_m, d_l);
    HIP_CHECK(hipGetLastError());

    // 异步D2H传输 - 到页锁定内存
    HIP_CHECK(hipMemcpyAsync(h_pinned_output, d_y, dataSize, hipMemcpyDeviceToHost, stream));
    
    // 等待所有异步操作完成
    HIP_CHECK(hipStreamSynchronize(stream));

    // 将结果从页锁定内存拷贝回用户提供的输出内存
    // 这是另一个快速的 host-to-host 拷贝
    memcpy(output, h_pinned_output, dataSize);

    // 清理所有资源
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_blockM));
    HIP_CHECK(hipFree(d_blockL));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_l));
    HIP_CHECK(hipHostFree(h_pinned_input));
    HIP_CHECK(hipHostFree(h_pinned_output));

    // 销毁流 - 注释掉，因为我们现在使用静态流，在程序结束时系统会自动清理
    // HIP_CHECK(hipStreamDestroy(stream));
}

// --- 基于在线法的 (m,l) 分块计算 ---
__global__ __launch_bounds__(TPB) void onlinePartialKernel(const float* __restrict__ x,
                                    float* __restrict__ blockM,
                                    double* __restrict__ blockL,
                                    int n) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float m_local = -FLT_MAX;
    double l_local = 0.0;

    // 向量化遍历 - 使用标准网格步长循环
    int numVec = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int idx4 = blockIdx.x * blockDim.x + tid;
    
    for (; idx4 < numVec; idx4 += stride) {
        float4 v4 = x4[idx4];
        // 依次合并四个元素
        combineRunningStats(m_local, l_local, v4.x, 1.0);
        combineRunningStats(m_local, l_local, v4.y, 1.0);
        combineRunningStats(m_local, l_local, v4.z, 1.0);
        combineRunningStats(m_local, l_local, v4.w, 1.0);
    }
    
    // 处理尾部不足4个元素
    int base = numVec << 2;
    int idx = base + blockIdx.x * blockDim.x + tid;
    for (; idx < n; idx += stride) {
        float vx = x[idx];
        combineRunningStats(m_local, l_local, vx, 1.0);
    }

    // 以两级归约方式合并到块级 (m,l)
    extern __shared__ unsigned char smemRaw[];
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    uintptr_t basePtr = reinterpret_cast<uintptr_t>(smemRaw);
    uintptr_t alignedPtr = (basePtr + 7u) & ~static_cast<uintptr_t>(7u);
    double* sL = reinterpret_cast<double*>(alignedPtr);
    float* sM = reinterpret_cast<float*>(sL + numWarps);

    // Warp 内归约
    float m_w = m_local;
    double l_w = l_local;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float m_o = __shfl_down(m_w, offset);
        double l_o = shfl_down_double(l_w, offset);
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
        float m_acc = -FLT_MAX;
        double l_acc = 0.0;
        if (lane < numWarps) {
            m_acc = sM[lane];
            l_acc = sL[lane];
        }
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float m_o = __shfl_down(m_acc, offset);
            double l_o = shfl_down_double(l_acc, offset);
            combineRunningStats(m_acc, l_acc, m_o, l_o);
        }
        if (lane == 0) {
            blockM[blockIdx.x] = m_acc;
            blockL[blockIdx.x] = l_acc;
        }
    }
}

// --- 最终归约得到 (m, l) ---
__global__ __launch_bounds__(TPB) void onlineFinalReduceKernel(const float* __restrict__ inM,
                                        const double* __restrict__ inL,
                                        float* __restrict__ outM,
                                        double* __restrict__ outL,
                                        int len) {
    const int tid = threadIdx.x;

    // 线程私有累积
    float m_local = -FLT_MAX;
    double l_local = 0.0;
    for (int i = tid; i < len; i += blockDim.x) {
        float m_i = inM[i];
        double l_i = inL[i];
        combineRunningStats(m_local, l_local, m_i, l_i);
    }

    // 两级归约
    extern __shared__ unsigned char smemRaw[];
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    uintptr_t basePtr = reinterpret_cast<uintptr_t>(smemRaw);
    uintptr_t alignedPtr = (basePtr + 7u) & ~static_cast<uintptr_t>(7u);
    double* sL = reinterpret_cast<double*>(alignedPtr);
    float* sM = reinterpret_cast<float*>(sL + numWarps);

    float m_w = m_local;
    double l_w = l_local;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        float m_o = __shfl_down(m_w, offset);
        double l_o = shfl_down_double(l_w, offset);
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
        float m_acc = -FLT_MAX;
        double l_acc = 0.0;
        if (lane < numWarps) {
            m_acc = sM[lane];
            l_acc = sL[lane];
        }
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            float m_o = __shfl_down(m_acc, offset);
            double l_o = shfl_down_double(l_acc, offset);
            combineRunningStats(m_acc, l_acc, m_o, l_o);
        }
        if (lane == 0) {
            *outM = m_acc;
            *outL = l_acc;
        }
    }
}

// --- 直接写出 y ---
__global__ __launch_bounds__(TPB) void softmaxWriteKernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int n,
                                   const float* __restrict__ mPtr,
                                   const double* __restrict__ lPtr) {
    const int stride = blockDim.x * gridDim.x;
    float m = *mPtr;
    double l = *lPtr;
    float denom = (float)fmax(1e-12, l);

    // 向量化归一化写出 - 使用标准网格步长循环
    int numVec = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (; idx4 < numVec; idx4 += stride) {
        float4 v4 = x4[idx4];
        v4.x = expf(v4.x - m) / denom;
        v4.y = expf(v4.y - m) / denom;
        v4.z = expf(v4.z - m) / denom;
        v4.w = expf(v4.w - m) / denom;
        y4[idx4] = v4;
    }
    
    // 处理尾部不足4个元素
    int base = numVec << 2;
    int idx = base + blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += stride) {
        y[idx] = expf(x[idx] - m) / denom;
    }
}
