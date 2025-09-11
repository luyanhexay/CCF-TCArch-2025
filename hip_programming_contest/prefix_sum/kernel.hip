// =========================
// kernel.hip  —  HIP 条带式（striped）前缀和（包含性）
//   - int 整数加法
//   - 一次块内写出“包含性”
//   - 两段式：块间用 blockSums 的 exclusive 前缀 + uniform_add
// =========================
#include "main.h"

// -------------------------
// 可调参数
// -------------------------
#ifndef BLOCK
#define BLOCK 512 // 需为 warpSize 的倍数（在 AMD 上 warpSize=64）
#endif

#ifndef ITEMS_PER_THREAD
#define ITEMS_PER_THREAD 8 // 每线程处理 K 个元素（推荐 4 或 8）
#endif

#ifndef CPU_FALLBACK_THRESHOLD
#define CPU_FALLBACK_THRESHOLD (0) // 小规模直接走 CPU（可按机型调）
#endif

#ifndef ENABLE_TIMING
#define ENABLE_TIMING 0
#endif

// -------------------------
// 错误检查宏
// -------------------------
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

struct GpuTimer
{
    hipEvent_t start{}, stop{};
    float ms{0.0f};
    GpuTimer()
    {
        hipEventCreate(&start);
        hipEventCreate(&stop);
    }
    ~GpuTimer()
    {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
    void tic() { hipEventRecord(start, 0); }
    float toc_sync()
    {
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

struct CpuTimer
{
    using clk = std::chrono::high_resolution_clock;
    std::chrono::time_point<clk> t0, t1;
    void tic() { t0 = clk::now(); }
    double toc_ms()
    {
        t1 = clk::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// -------------------------
// 简单 CPU 包含性前缀（用于小 N fallback / 或校验）
// -------------------------
static inline void cpu_inclusive_scan(const int *in, int *out, int n)
{
    long long acc = 0; // 用更宽以避免大 N 溢出；本题是 int，若需可再截断
    for (int i = 0; i < n; ++i)
    {
        acc += in[i];
        out[i] = static_cast<int>(acc);
    }
}

// -------------------------
// warp 内包含性扫描（整型），返回包含性
//   - 使用 HIP 的 __shfl_up
//   - 兼容 AMD（warpSize=64）与 NV（warpSize=32）
// -------------------------
__device__ __forceinline__ int warp_inclusive_scan_int(int x)
{
    int w = warpSize;
    // 经典二进制间距的分层前缀
    for (int off = 1; off < w; off <<= 1)
    {
        int y = __shfl_up(x, off, w); // 读取 lane-off 的值（越界时保持自身）
        if ((int)(threadIdx.x % w) >= off)
            x += y;
    }
    return x; // 包含性
}

// ===================================================
// 条带式块内前缀（包含性写回）
// ---------------------------------------------------
// 每块处理 TILE = BLOCK * K 个元素
// 全局地址映射： idx = base + tid + k*BLOCK
// 步骤：
//   A) 线程内串行小前缀：vals[k] = Σ_{j<=k} a(tid,j)
//   B) 对每个切片 k：warp 内对 a(tid,k) 做包含性扫描，取 exclusive 加回到 vals[k]
//   C) 对每个切片 k：warp 间（numWarps<=warpSize）做包含性→exclusive，广播加回 vals[k]
//   D) 一次性写回（包含性）；写出 blockSum（本块最后一个有效元素的包含性值）
// 共享内存：K * numWarps * sizeof(int)
// ===================================================
template <int K>
__global__ void block_scan_striped_inclusive(
    const int *__restrict__ in,
    int *__restrict__ out,
    int *__restrict__ blockSums,
    int N)
{
    // 共享内存布局：
    // [0 .. K*numWarps-1] : 每列(k)×每warp 的临时和/跨warp基准
    // [K*numWarps]        : col_prefix（更早列的块级总和）
    // [K*numWarps + 1]    : col_total  （当前列的块级总和）
    extern __shared__ int smem[];
    const int numWarps = BLOCK / warpSize;

    int *warp_sums = smem;
    // 使用 volatile 引用，避免编译器把共享变量缓存穿越 __syncthreads()
    volatile int &col_prefix = *(volatile int *)&smem[K * numWarps + 0];
    volatile int &col_total = *(volatile int *)&smem[K * numWarps + 1];

    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int wid = tid / warpSize;
    const int bid = blockIdx.x;

    const int TILE = BLOCK * K;
    const int base = bid * TILE;

    // ---- 初始化块级列前缀 ----
    if (tid == 0)
    {
        col_prefix = 0;
        col_total = 0;
    }
    __syncthreads();

    // 线程私有：原始值 a(tid,k) 与“仅本线程的小前缀”
    int ak[K];
    int vals[K]; // 仅本线程：vals[k] = Σ_{j<=k} a(tid,j)
#pragma unroll
    for (int k = 0; k < K; ++k)
    {
        int idx = base + tid + k * BLOCK; // 条带式
        int v = (idx < N) ? in[idx] : 0;
        ak[k] = v;
        vals[k] = (k == 0) ? v : (vals[k - 1] + v);
    }

    // 动态维护：本线程对“更早列”的和 = Σ_{j<k} a(tid,j)
    int thread_prefix_running = 0;

// ---- 逐列处理：本列跨线程 + 更早列块前缀 ----
#pragma unroll
    for (int k = 0; k < K; ++k)
    {
        // 1) 本列在“同 warp 内”的包含性前缀（对 ak[k]）
        int incl = ak[k];
        for (int off = 1; off < warpSize; off <<= 1)
        {
            int y = __shfl_up(incl, off, warpSize);
            if (lane >= off)
                incl += y;
        }
        int excl_intra = incl - ak[k]; // 同 warp 内 exclusive

        // 把每个 warp 的该列合计写到共享（由该 warp 的最后 lane 写）
        if (lane == warpSize - 1)
            warp_sums[k * numWarps + wid] = incl;
        __syncthreads();

        // 2) “跨 warp”的 exclusive 基准 + 本列总和 col_total
        if (wid == 0)
        {
            // 取 numWarps 个值，其他 lane 取 0
            int v = (lane < numWarps) ? warp_sums[k * numWarps + lane] : 0;

            // warp0 内做包含性
            int vincl = v;
            for (int off = 1; off < warpSize; off <<= 1)
            {
                int y = __shfl_up(vincl, off, warpSize);
                if (lane >= off)
                    vincl += y;
            }

            // 写回“跨 warp 的 exclusive 基准”
            if (lane < numWarps)
            {
                int v_excl = (lane == 0) ? 0 : (vincl - warp_sums[k * numWarps + lane]);
                warp_sums[k * numWarps + lane] = v_excl;
            }

            // lane==numWarps-1 负责把“本列总和”放进共享标量
            if (lane == numWarps - 1)
                col_total = vincl;
        }
        __syncthreads();

        const int warp_base = warp_sums[k * numWarps + wid]; // 本列跨warp基准（exclusive）
        const int col_pref = col_prefix;                     // 之前所有列的块和
        __syncthreads();

        // 3) 组装本列最终包含性值：
        //    out = col_pref  +  Σ_{u<tid} a(u,k)  +  a(tid,k)
        //          ^^^^^^^^^     ^^^^^^^^^^^^^^^     ^^^^^^^^
        //    其中 Σ_{u<tid} a(u,k) = excl_intra + warp_base；
        //    vals[k] = Σ_{j<=k} a(tid,j)（仅本线程），所以需要补 (col_pref - Σ_{j<k} a(tid,j))：
        vals[k] += (excl_intra + warp_base) + (col_pref - thread_prefix_running);

        // 4) 列末：由 tid==0 更新“列前缀”供下一列使用
        if (tid == 0)
            col_prefix = col_pref + col_total;
        __syncthreads();

        // 维护“本线程更早列之和”
        thread_prefix_running += ak[k];
    }

// ---- 一次性写回（包含性） ----
#pragma unroll
    for (int k = 0; k < K; ++k)
    {
        int idx = base + tid + k * BLOCK;
        if (idx < N)
            out[idx] = vals[k];
    }

    // ---- 写 blockSum（本块最后一个有效元素的包含性值）----
    if (blockSums)
    {
        int last = min(TILE, N - base) - 1;
        if (last >= 0 && tid == (last % BLOCK))
        {
            int k_last = last / BLOCK;
            blockSums[bid] = vals[k_last];
        }
    }
}

// ---------------------------------------------------
// uniform add：把 exclusive 块偏移加回每块的所有元素
//  注意：blockOffsets[bid] 必须是 blockSums 的 exclusive 扫描结果
// ---------------------------------------------------
template <int K>
__global__ void uniform_add_striped(
    const int *__restrict__ blockOffsets, // exclusive: Σ_{p<bid} blockSum[p]
    int *__restrict__ out,
    int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int TILE = BLOCK * K;
    const int base = bid * TILE;
    const int addv = blockOffsets[bid]; // 第 0 块自然为 0

#pragma unroll
    for (int k = 0; k < K; ++k)
    {
        int idx = base + tid + k * BLOCK;
        if (idx < N)
            out[idx] += addv;
    }
}

// ===================================================
// 入口：保持接口不变
//   - 默认输出“包含性”前缀
// ===================================================
extern "C" void solve(const int *input, int *output, int N)
{
    // if (N <= 0)
    //     return;

    // // 小规模直接 CPU：避免 H2D/D2H 与 kernel 启动开销
    // if (N <= CPU_FALLBACK_THRESHOLD)
    // {
    //     cpu_inclusive_scan(input, output, N);
    //     return;
    // }
#if ENABLE_TIMING
    CpuTimer cpuAll;
    cpuAll.tic();
#endif

    constexpr int K = ITEMS_PER_THREAD;
    const int TILE = BLOCK * K;
    const int numBlocks = (N + TILE - 1) / TILE;

    // ---------- 计时器 ----------
#if ENABLE_TIMING
    GpuTimer tH2D, tK_block, tBS_D2H, tBS_scanCPU, tBS_H2D, tK_add, tD2H, tFree, tSync;
    CpuTimer tAlloc, tBS_cpuScan, tAllHostOps;
    tAllHostOps.tic();
#endif

    // ---------- 分配 ----------
#if ENABLE_TIMING
    tAlloc.tic();
#endif

    // 设备内存
    int *d_in = nullptr, *d_out = nullptr, *d_bs = nullptr, *d_bs_excl = nullptr;
    HIP_CHECK(hipMalloc(&d_in, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    // HIP_CHECK(hipMemcpy(d_in, input, N * sizeof(int), hipMemcpyHostToDevice));

    if (numBlocks > 1)
    {
        HIP_CHECK(hipMalloc(&d_bs, numBlocks * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_bs_excl, numBlocks * sizeof(int)));
    }
#if ENABLE_TIMING
    double msAlloc = tAlloc.toc_ms();
#endif
    // ---------- H2D ----------
#if ENABLE_TIMING
    tH2D.tic();
#endif
    HIP_CHECK(hipMemcpy(d_in, input, N * sizeof(int), hipMemcpyHostToDevice));
#if ENABLE_TIMING
    float msH2D = tH2D.toc_sync();
#endif
#if ENABLE_TIMING
    tK_block.tic();
#endif

    // 共享内存：K * numWarps * sizeof(int)
    const int numWarps = BLOCK / warpSize;                                // AMD: warpSize=64
    size_t shmem_bytes = (ITEMS_PER_THREAD * numWarps + 2) * sizeof(int); // +2: col_prefix 与 col_total_s

    // 1) 块内条带式前缀（包含性写回） + 写出 blockSums
    hipLaunchKernelGGL((block_scan_striped_inclusive<K>),
                       dim3(numBlocks), dim3(BLOCK), shmem_bytes, 0,
                       d_in, d_out, d_bs, N);
    HIP_CHECK(hipGetLastError());
#if ENABLE_TIMING
    float msK_block = tK_block.toc_sync();
#endif
// ---------- 块间 exclusive（两段式） ----------
#if ENABLE_TIMING
    float msBS_D2H = 0, msBS_H2D = 0, msK_add_ms = 0, msBS_cpuScan = 0;
    double msCpuBlockScan = 0.0;
#endif
    // 2) 块间：对 blockSums 做 exclusive 扫描，再做 uniform add
    if (numBlocks > 1)
    {
#if ENABLE_TIMING
        tBS_D2H.tic();
#endif
        // 简易路线：拷到 CPU 做 exclusive（可替换为 GPU 递归）
        std::vector<int> h_bs(numBlocks), h_ex(numBlocks);
        HIP_CHECK(hipMemcpy(h_bs.data(), d_bs, numBlocks * sizeof(int), hipMemcpyDeviceToHost));
#if ENABLE_TIMING
        msBS_D2H = tBS_D2H.toc_sync();
#endif
#if ENABLE_TIMING
        tBS_cpuScan.tic();
#endif
        long long acc = 0;
        for (int i = 0; i < numBlocks; ++i)
        {
            h_ex[i] = static_cast<int>(acc);
            acc += h_bs[i];
        }
#if ENABLE_TIMING
        msCpuBlockScan = tBS_cpuScan.toc_ms();
#endif
        // --- H2D: blockOffsets (exclusive) ---
#if ENABLE_TIMING
        tBS_H2D.tic();
#endif
        HIP_CHECK(hipMemcpy(d_bs_excl, h_ex.data(), numBlocks * sizeof(int), hipMemcpyHostToDevice));
#if ENABLE_TIMING
        msBS_H2D = tBS_H2D.toc_sync();
#endif
        // uniform add（条带式索引保持一致）
#if ENABLE_TIMING
        tK_add.tic();
#endif
        hipLaunchKernelGGL((uniform_add_striped<K>),
                           dim3(numBlocks), dim3(BLOCK), 0, 0,
                           d_bs_excl, d_out, N);
        HIP_CHECK(hipGetLastError());
#if ENABLE_TIMING
        msK_add_ms = tK_add.toc_sync();
#endif
    }

// 3) 回拷
#if ENABLE_TIMING
    tD2H.tic();
#endif
    HIP_CHECK(hipMemcpy(output, d_out, N * sizeof(int), hipMemcpyDeviceToHost));
#if ENABLE_TIMING
    float msD2H = tD2H.toc_sync();
#endif
    // ---------- 同步（确保所有 GPU 活动结束） ----------
#if ENABLE_TIMING
    tSync.tic();
#endif
    HIP_CHECK(hipDeviceSynchronize());
#if ENABLE_TIMING
    float msSync = tSync.toc_sync(); // 这里一般很小
#endif
    // 4) 资源回收
#if ENABLE_TIMING
    tFree.tic();
#endif
    if (d_bs_excl)
        hipFree(d_bs_excl);
    if (d_bs)
        hipFree(d_bs);
    hipFree(d_out);
    hipFree(d_in);
#if ENABLE_TIMING
    float msFreeSync = tFree.toc_sync();
    double msHostOps = tAllHostOps.toc_ms();
#endif
#if ENABLE_TIMING
    double msAll = cpuAll.toc_ms();
    // 分项打印
    printf("[TIMING] solve(): N=%d, BLOCK=%d, K=%d, numBlocks=%d\n", N, BLOCK, K, numBlocks);
    printf("  Alloc(HOST+API)=%.3f ms\n", msAlloc);
    printf("  H2D          =%.3f ms\n", msH2D);
    printf("  Kern-Block   =%.3f ms\n", msK_block);
    if (numBlocks > 1)
    {
        printf("  BS D2H       =%.3f ms\n", msBS_D2H);
        printf("  BS CPU scan  =%.3f ms\n", msCpuBlockScan);
        printf("  BS H2D       =%.3f ms\n", msBS_H2D);
        printf("  Kern-Add     =%.3f ms\n", msK_add_ms);
    }
    printf("  D2H          =%.3f ms\n", msD2H);
    printf("  DevSync      =%.3f ms\n", msSync);
    printf("  Free         =%.3f ms\n", msFreeSync);
    printf("  HostOps(all in solve) = %.3f ms\n", msHostOps);
    printf("  === TOTAL solve wall time = %.3f ms ===\n", msAll);
#endif
}
