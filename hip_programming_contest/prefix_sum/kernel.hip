#include "main.h"

/* write your code here*/
#ifndef BLOCK
#define BLOCK 256
#endif

__global__ void scan_block_2elems(const int *in, int *out, int *blockSums,
                                  int N)
{
    __shared__ int s[2 * BLOCK];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int base = bid * BLOCK * 2;

    // 计算出在完整数据上的位置
    int i0 = base + 2 * tid;
    int i1 = i0 + 1;
    // 这是相对位置，在块内的位置
    s[2 * tid] = (i0 < N) ? in[i0] : 0;
    s[2 * tid + 1] = (i1 < N) ? in[i1] : 0;

    __syncthreads();
    // 这里先计算出奇数位置的前缀和（只是下标是奇数，存的还是对于整个数组的前缀和）
    for (int offset = 1; offset < 2 * BLOCK; offset <<= 1)
    {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < 2 * BLOCK)
        {
            s[idx] += s[idx - offset];
        }
        __syncthreads();
    }
    // 在主线程把当前块的总和存到blockSums里
    if (tid == 0)
    {
        if (blockSums)
            blockSums[bid] = s[2 * BLOCK - 1];
        s[2 * BLOCK - 1] = 0;
    }
    __syncthreads();
    // 有待理解
    for (int offset = (2 * BLOCK) >> 1; offset > 0; offset >>= 1)
    {
        int idx = (tid + 1) * (offset << 1) - 1;
        if (idx < 2 * BLOCK)
        {
            int t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }
    if (i0 < N)
        out[i0] = s[2 * tid] + in[i0];
    if (i1 < N)
        out[i1] = s[2 * tid + 1] + in[i1];
}

__global__ void add_block_offsets(int *out, const int *offsets, int N)
{
    const int bid = blockIdx.x;
    const int base = bid * BLOCK * 2;
    const int tid = threadIdx.x;
    int i0 = base + 2 * tid;
    int i1 = i0 + 1;
    int offset = offsets[bid];
    if (i0 < N)
        out[i0] += offset;
    if (i1 < N)
        out[i1] += offset;
}
__global__ void inclusive_to_exclusive_inplace(int *a, int n)
{
    // 单线程顺序右移：a[i] = 累积到 i-1 的和；a[0] = 0
    int prev = 0;
    for (int i = 0; i < n; ++i)
    {
        int cur = a[i];
        a[i] = prev;
        prev = cur;
    }
}

extern "C" void solve(const int *input, int *output, int N)
{
    /* write your code here*/
    // 1) 基本参数：一段(segment)长度=每块处理的元素数=2*BLOCK（你的核就是这么设计的）
    const int SEG = 2 * BLOCK;
    const int numSeg = (N + SEG - 1) / SEG; // 段的数量（向上取整）

    // 2) 设备端（GPU）内存指针
    int *d_in = nullptr;
    int *d_out = nullptr;
    int *d_blockSums = nullptr; // 存每段的总和；稍后会把它变成“每段的独占偏移量”

    // 3) 在 GPU 上分配显存
    hipMalloc(&d_in, N * sizeof(int));  // 输入数组
    hipMalloc(&d_out, N * sizeof(int)); // 输出数组（各段的前缀和 + 最终结果）
    if (numSeg > 0)
        hipMalloc(&d_blockSums, numSeg * sizeof(int)); // 各段总和/偏移

    // 4) 把输入从主机(CPU)拷到设备(GPU)
    hipMemcpy(d_in, input, N * sizeof(int), hipMemcpyHostToDevice);

    // 5) 发射“段内前缀和”内核（你的 scan_block_2elems）
    //    grid=段数；block=BLOCK；共享内存用静态数组，不需要额外字节
    if (numSeg > 0)
    {
        dim3 grid(numSeg), block(BLOCK);
        hipLaunchKernelGGL(scan_block_2elems,
                           grid, block,
                           0, 0, // sharedMemBytes=0（你用的是静态shared数组），stream=0
                           d_in, d_out, d_blockSums, N);
    }

    // 6) 计算“跨段偏移”：对 d_blockSums 做一次前缀和并转成 exclusive
    //    为了简单稳妥：
    //    - 如果段数不大（<= SEG），直接用“同一段扫描核”在 GPU 上一口气扫完；
    //    - 如果段数很大，就走 CPU 路径（仍然正确，只是偏慢；比赛里够用）。
    if (numSeg > 0)
    {
        if (numSeg <= SEG)
        {
            // 6a) GPU 路径：用你的 scan_block_2elems 在单块内把 blockSums 扫成 inclusive
            dim3 grid1(1), block1(BLOCK);
            hipLaunchKernelGGL(scan_block_2elems,
                               grid1, block1,
                               0, 0,
                               d_blockSums, // in
                               d_blockSums, // out（就地）
                               nullptr,     // 不需要再收集“段和里的段和”
                               numSeg);     // n=段数

            // 6b) 把 inclusive 变 exclusive（段偏移 = 前面所有段之和，不含自己）
            hipLaunchKernelGGL(inclusive_to_exclusive_inplace,
                               dim3(1), dim3(1),
                               0, 0,
                               d_blockSums, numSeg);
        }
        else
        {
            // 6c) CPU 路径（简洁可靠）：把 blockSums 拷回主机做独占前缀和，再拷回设备
            std::vector<int> h_blockSums(numSeg);
            hipMemcpy(h_blockSums.data(), d_blockSums, numSeg * sizeof(int), hipMemcpyDeviceToHost);

            int run = 0;
            for (int i = 0; i < numSeg; ++i)
            {
                int v = h_blockSums[i];
                h_blockSums[i] = run; // 写入独占偏移
                run += v;
            }
            hipMemcpy(d_blockSums, h_blockSums.data(), numSeg * sizeof(int), hipMemcpyHostToDevice);
        }
    }

    // 7) 发射“加偏移”内核（你的 add_block_offsets）
    if (numSeg > 0)
    {
        dim3 grid(numSeg), block(BLOCK);
        hipLaunchKernelGGL(add_block_offsets,
                           grid, block,
                           0, 0,
                           d_out, d_blockSums, N);
    }

    // 8) 结果拷回主机
    hipMemcpy(output, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

    // 9) 释放显存
    hipFree(d_in);
    hipFree(d_out);
    if (d_blockSums)
        hipFree(d_blockSums);
}