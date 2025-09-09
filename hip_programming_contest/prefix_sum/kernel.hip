// kernel.hip
#include "main.h"

// =========================
// 配置
// =========================
#ifndef BLOCK
#define BLOCK 256
#endif

// 是否在 solve 中做 GPU kernel 用时打印
#ifndef ENABLE_KERNEL_TIMING
#define ENABLE_KERNEL_TIMING 1
#endif

// 小规模时直接在 CPU 上完成整段扫描（避免 GPU 启动开销）
// 阈值可按机器调，单位：元素个数
#ifndef CPU_FALLBACK_THRESHOLD
#define CPU_FALLBACK_THRESHOLD (1 << 13) // 16384
#endif

// 错误检查
#define HIP_CHECK(cmd)                                                                          \
    do                                                                                          \
    {                                                                                           \
        hipError_t e = (cmd);                                                                   \
        if (e != hipSuccess)                                                                    \
        {                                                                                       \
            fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            std::abort();                                                                       \
        }                                                                                       \
    } while (0)

// =========================
// 块内扫描核：Blelloch exclusive
// - 每块处理 2*BLOCK 个元素
// - 一线程两元素
// - 写出块内 exclusive 结果到 out
// - 并把本块总和写到 blockSums[bid]
// =========================
__global__ void scan_block_2elems(const int *__restrict__ in,
                                  int *__restrict__ out,
                                  int *__restrict__ blockSums,
                                  int N)
{
    __shared__ int s[2 * BLOCK];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int base = bid * BLOCK * 2;

    // 本块有效元素数 n
    const int n = max(0, min(2 * BLOCK, N - base));
    if (n <= 0)
    {
        if (blockSums && tid == 0)
            blockSums[bid] = 0;
        return;
    }

    // 计算 m = nextPow2(n)（建满二叉树）
    int m = 1;
    while (m < n)
        m <<= 1; // m <= 2*BLOCK 恒成立

    // 全局位置
    const int i0 = base + tid;
    const int i1 = base + BLOCK + tid;

    // 读两元素（越界读 0）
    int v0 = (i0 < N) ? in[i0] : 0;
    int v1 = (i1 < N) ? in[i1] : 0;

    // 写共享内存；对 [n, m) 做 0 填充，避免后续访问未初始化数据
    int j0 = tid;
    int j1 = tid + BLOCK;

    if (j0 < n)
        s[j0] = v0;
    else if (j0 < m)
        s[j0] = 0;

    if (j1 < n)
        s[j1] = v1;
    else if (j1 < m)
        s[j1] = 0;

    __syncthreads();

    // ---- upsweep：用 m（2 的幂）建树 ----
    for (int offset = 1; offset < m; offset <<= 1)
    {
        __syncthreads();
        int idx = ((tid + 1) << 1) * offset - 1;
        if (idx < m)
            s[idx] += s[idx - offset];
    }
    __syncthreads();

    // 总和在 s[m-1]；写出块和，并把 s[m-1] 清零作为 exclusive 根
    if (tid == 0)
    {
        if (blockSums)
            blockSums[bid] = s[m - 1];
        s[m - 1] = 0;
    }
    __syncthreads();

    // ---- downsweep：同样用 m ----
    for (int offset = (m >> 1); offset >= 1; offset >>= 1)
    {
        __syncthreads();
        int idx = ((tid + 1) << 1) * offset - 1;
        if (idx < m)
        {
            int t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
    }
    __syncthreads();

    // 写回：只写有效范围
    if (i0 < N)
        out[i0] = s[tid];
    if (i1 < N)
        out[i1] = s[tid + BLOCK];
}

// =========================
// 跨块回加：把 blockSums 的 exclusive 前缀加回 out
// =========================
__global__ void uniform_add(const int *__restrict__ blockSumsScan,
                            int *__restrict__ out,
                            int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = bid * BLOCK * 2;

    if (bid == 0)
        return; // 第 0 块偏移为 0

    const int addv = blockSumsScan[bid];

    const int i0 = base + tid;
    const int i1 = base + BLOCK + tid;

    if (i0 < N)
        out[i0] += addv;
    if (i1 < N)
        out[i1] += addv;
}
// =========================
// 简单的 CPU exclusive scan（用于 blockSums）
// in/out 可以相同
// =========================
static void cpu_exclusive_scan(const int *in, int *out, int n)
{
    int acc = 0;
    for (int i = 0; i < n; ++i)
    {
        int v = in[i];
        out[i] = acc;
        acc += v;
    }
}
static void cpu_inclusive_scan(const int *in, int *out, int n)
{
    int acc = 0;
    for (int i = 0; i < n; ++i)
    {
        acc += in[i];
        out[i] = acc;
    }
}

__global__ void to_inclusive(const int *__restrict__ in,
                             int *__restrict__ out,
                             int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] += in[i];
}

// =========================
// 入口：保持接口不变
// - 默认输出 exclusive 前缀和（与上面核一致）
// =========================
void solve(const int *input, int *output, int N)
{
    if (N <= 0)
        return;

    // 小规模直接 CPU（避免 GPU 启动/拷贝开销）
    if (N <= CPU_FALLBACK_THRESHOLD)
    {
        cpu_inclusive_scan(input, output, N);
        return;
    }

    const int elems_per_block = 2 * BLOCK;
    const int numBlocks = (N + elems_per_block - 1) / elems_per_block;

    // 1) 设备内存
    int *d_in = nullptr, *d_out = nullptr, *d_blockSums = nullptr;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    if (numBlocks > 1)
        HIP_CHECK(hipMalloc(&d_blockSums, numBlocks * sizeof(int)));

    // 2) 把 host 指针注册为 page-locked（加速 H2D/D2H）
    //    注意：如果 input/output 来自栈或临时内存，请确保生命周期覆盖整个 solve
    bool pinned_in = false, pinned_out = false;
    hipError_t r_in = hipHostRegister(const_cast<int *>(input), N * sizeof(int), hipHostRegisterDefault);
    if (r_in == hipSuccess || r_in == hipErrorHostMemoryAlreadyRegistered)
        pinned_in = true;

    hipError_t r_out = hipHostRegister(output, N * sizeof(int), hipHostRegisterDefault);
    if (r_out == hipSuccess || r_out == hipErrorHostMemoryAlreadyRegistered)
        pinned_out = true;

    // 3) H2D
    HIP_CHECK(hipMemcpy(d_in, input, N * sizeof(int), hipMemcpyHostToDevice));

    // 4) 计时（仅核时间）
#if ENABLE_KERNEL_TIMING
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, 0));
#endif

    // 5) 段内扫描
    dim3 grid(numBlocks), block(BLOCK);
    if (numBlocks > 0)
    {
        hipLaunchKernelGGL(scan_block_2elems, grid, block, 0, 0,
                           d_in, d_out, d_blockSums, N);
        HIP_CHECK(hipGetLastError());
    }

    // 6) 处理 blockSums：在 CPU 做 exclusive，再回加
    if (numBlocks > 1)
    {
        std::vector<int> h_blockSums(numBlocks, 0);
        std::vector<int> h_blockSumsScan(numBlocks, 0);

        // 拷回块和
        HIP_CHECK(hipMemcpy(h_blockSums.data(), d_blockSums,
                            numBlocks * sizeof(int), hipMemcpyDeviceToHost));

        // CPU exclusive
        cpu_exclusive_scan(h_blockSums.data(), h_blockSumsScan.data(), numBlocks);

        // 传回设备
        HIP_CHECK(hipMemcpy(d_blockSums, h_blockSumsScan.data(),
                            numBlocks * sizeof(int), hipMemcpyHostToDevice));

        // 回加
        hipLaunchKernelGGL(uniform_add, grid, block, 0, 0, d_blockSums, d_out,
                           N);
        HIP_CHECK(hipGetLastError());
    }
    {
        dim3 blk(BLOCK);
        dim3 grd((N + BLOCK - 1) / BLOCK);
        hipLaunchKernelGGL(to_inclusive, grd, blk, 0, 0, d_in, d_out, N);
        HIP_CHECK(hipGetLastError());
    }

#if ENABLE_KERNEL_TIMING
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    // printf("Kernel time: %.3f ms\n", ms);
    HIP_CHECK(hipEventDestroy(start));

    HIP_CHECK(hipEventDestroy(stop));
#endif

    // 7) D2H
    HIP_CHECK(hipMemcpy(output, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    // 8) 释放/反注册
    if (pinned_in)
        hipHostUnregister(const_cast<int *>(input));
    if (pinned_out)
        hipHostUnregister(output);

    if (d_blockSums)
        hipFree(d_blockSums);
    hipFree(d_out);
    hipFree(d_in);
}
