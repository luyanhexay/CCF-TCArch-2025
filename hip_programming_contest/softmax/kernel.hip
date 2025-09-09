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

__global__ void reduceMaxKernel(const float* __restrict__ x, float* __restrict__ blockMax, int n) {
    extern __shared__ float sdataF[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float localMax = -FLT_MAX;
    for (; idx < n; idx += stride) {
        float v = x[idx];
        localMax = v > localMax ? v : localMax;
    }

    sdataF[tid] = localMax;
    __syncthreads();

    // 共享内存内树形归约
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = sdataF[tid + offset];
            sdataF[tid] = sdataF[tid] > other ? sdataF[tid] : other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMax[blockIdx.x] = sdataF[0];
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
    for (; idx < n; idx += stride) {
        float shifted = x[idx] - m;
        double t = (double)expf(shifted);
        y[idx] = (float)t; // 缓存 t_i，避免后续重复计算 exp
        localSum += t;
    }

    sdataD[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdataD[tid] += sdataD[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = sdataD[0];
    }
}

__global__ void softmaxNormalizeKernel(float* __restrict__ y, int n, const double* __restrict__ sumExpPtr) {
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double sumExp = *sumExpPtr;
    float denom = (float)fmax(1e-12, sumExp);
    for (; idx < n; idx += stride) {
        y[idx] = y[idx] / denom;
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

    sdataF[tid] = localMax;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = sdataF[tid + offset];
            sdataF[tid] = sdataF[tid] > other ? sdataF[tid] : other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        *outVal = sdataF[0];
    }
}

__global__ void reduceFinalSumKernel(const double* __restrict__ inVals, double* __restrict__ outVal, int len) {
    extern __shared__ double sdataD[];
    const int tid = threadIdx.x;

    double localSum = 0.0;
    for (int i = tid; i < len; i += blockDim.x) {
        localSum += inVals[i];
    }

    sdataD[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdataD[tid] += sdataD[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *outVal = sdataD[0];
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

    // 1) 求全局最大值 m
    float* d_blockMax = nullptr;
    HIP_CHECK(hipMalloc(&d_blockMax, sizeof(float) * (size_t)blocks));
    size_t smemMax = sizeof(float) * (size_t)threads;
    reduceMaxKernel<<<blocks, threads, smemMax>>>(d_x, d_blockMax, N);
    HIP_CHECK(hipGetLastError());

    // 设备端最终归约得到 m
    float* d_m = nullptr;
    HIP_CHECK(hipMalloc(&d_m, sizeof(float)));
    reduceFinalMaxKernel<<<1, threads, smemMax>>>(d_blockMax, d_m, blocks);
    HIP_CHECK(hipGetLastError());

    // 2) 计算 sum(exp(x - m))
    double* d_blockSums = nullptr;
    HIP_CHECK(hipMalloc(&d_blockSums, sizeof(double) * (size_t)blocks));
    size_t smemSum = sizeof(double) * (size_t)threads;
    reduceSumExpKernel<<<blocks, threads, smemSum>>>(d_x, d_y, d_blockSums, N, d_m);
    HIP_CHECK(hipGetLastError());

    // 设备端最终归约得到 sumExp
    double* d_sumExp = nullptr;
    HIP_CHECK(hipMalloc(&d_sumExp, sizeof(double)));
    reduceFinalSumKernel<<<1, threads, smemSum>>>(d_blockSums, d_sumExp, blocks);
    HIP_CHECK(hipGetLastError());

    // 3) 归一化输出
    softmaxNormalizeKernel<<<blocks, threads>>>(d_y, N, d_sumExp);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(output, d_y, sizeof(float) * (size_t)N, hipMemcpyDeviceToHost));

    // 清理资源
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_blockMax));
    HIP_CHECK(hipFree(d_blockSums));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_sumExp));
    HIP_CHECK(hipHostUnregister((void*)input));
    HIP_CHECK(hipHostUnregister((void*)output));
}