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

__global__ void reduceSumExpKernel(const float* __restrict__ x, double* __restrict__ blockSums, int n, float m) {
    extern __shared__ double sdataD[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double localSum = 0.0;
    for (; idx < n; idx += stride) {
        float shifted = x[idx] - m;
        // 数值稳定：shifted <= 0
        double t = (double)expf(shifted);
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

__global__ void softmaxNormalizeKernel(const float* __restrict__ x, float* __restrict__ y, int n, float m, double sumExp) {
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float denom = (float)fmax(1e-12, sumExp);
    for (; idx < n; idx += stride) {
        float shifted = x[idx] - m;
        float t = expf(shifted);
        y[idx] = t / denom;
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

    HIP_CHECK(hipMemcpy(d_x, input, sizeof(float) * (size_t)N, hipMemcpyHostToDevice));

    // 1) 求全局最大值 m
    float* d_blockMax = nullptr;
    HIP_CHECK(hipMalloc(&d_blockMax, sizeof(float) * (size_t)blocks));
    size_t smemMax = sizeof(float) * (size_t)threads;
    reduceMaxKernel<<<blocks, threads, smemMax>>>(d_x, d_blockMax, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // 拷回分块最大值并在主机端做最终归约
    std::vector<float> h_blockMax(blocks);
    HIP_CHECK(hipMemcpy(h_blockMax.data(), d_blockMax, sizeof(float) * (size_t)blocks, hipMemcpyDeviceToHost));
    float m = -FLT_MAX;
    for (int i = 0; i < blocks; ++i) m = h_blockMax[i] > m ? h_blockMax[i] : m;

    // 2) 计算 sum(exp(x - m))
    double* d_blockSums = nullptr;
    HIP_CHECK(hipMalloc(&d_blockSums, sizeof(double) * (size_t)blocks));
    size_t smemSum = sizeof(double) * (size_t)threads;
    reduceSumExpKernel<<<blocks, threads, smemSum>>>(d_x, d_blockSums, N, m);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> h_blockSums(blocks);
    HIP_CHECK(hipMemcpy(h_blockSums.data(), d_blockSums, sizeof(double) * (size_t)blocks, hipMemcpyDeviceToHost));
    double sumExp = 0.0;
    for (int i = 0; i < blocks; ++i) sumExp += h_blockSums[i];

    // 3) 归一化输出
    softmaxNormalizeKernel<<<blocks, threads>>>(d_x, d_y, N, m, sumExp);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output, d_y, sizeof(float) * (size_t)N, hipMemcpyDeviceToHost));

    // 清理资源
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_blockMax));
    HIP_CHECK(hipFree(d_blockSums));
}