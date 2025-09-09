#include "main.h"
#include <cstdlib>
#include <algorithm>

// 常量：不可达的距离值
static constexpr int INF = 1073741823; // 2^30 - 1

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(e) << "\n"; \
        std::exit(1); \
    } \
} while(0)

__device__ __forceinline__ int sat_add_if_valid(int a, int b) {
    if (a >= INF || b >= INF) return INF;
    long long s = (long long)a + (long long)b;
    if (s > (long long)INF) s = (long long)INF;
    return (int)s;
}

// Phase 1: Update pivot tile (t, t)
__global__ void fw_phase1_pivot(int* __restrict__ D, int V, int B, int t) {
    extern __shared__ int smem[]; // size >= B*(B+1)
    int* sP = smem;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int base = t * B;
    const int tile = (V - base) < B ? (V - base) : B;

    const int stride = B + 1; // padding to mitigate bank conflicts
    // Load pivot tile into shared（标量）
    if (ty < tile && tx < tile) {
        sP[ty * stride + tx] = D[(base + ty) * V + (base + tx)];
    }
    __syncthreads();

    // In-place FW within the tile
    for (int k = 0; k < tile; ++k) {
        __syncthreads();
        if (ty < tile && tx < tile) {
            int dij = sP[ty * stride + tx];
            int dik = sP[ty * stride + k];
            int dkj = sP[k * stride + tx];
            int alt = sat_add_if_valid(dik, dkj);
            if (alt < dij) sP[ty * stride + tx] = alt;
        }
    }
    __syncthreads();

    // Store back（标量）
    if (ty < tile && tx < tile) {
        D[(base + ty) * V + (base + tx)] = sP[ty * stride + tx];
    }
}

// Phase 2 (row): Update tiles (t, j) for all j != t
__global__ void fw_phase2_row(int* __restrict__ D, int V, int B, int t, int nTiles) {
    extern __shared__ int smem[]; // sP (B*(B+1)) + sR (B*(B+1)) + rowK (B)
    const int stride = B + 1;
    int* sP    = smem;
    int* sR    = sP + B * stride;
    int* rowK = sR + B * stride;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int jTile = blockIdx.x;
    if (jTile >= nTiles - 1) return;
    const int j = (jTile < t) ? jTile : (jTile + 1);

    const int baseT = t * B;
    const int baseJ = j * B;
    const int pivot = (V - baseT) < B ? (V - baseT) : B;
    const int w     = (V - baseJ) < B ? (V - baseJ) : B;

    // Load pivot tile (pivot x pivot)
    if (ty < pivot && tx < pivot) {
        sP[ty * stride + tx] = D[(baseT + ty) * V + (baseT + tx)];
    }
    // Load row tile (pivot x w)
    if (ty < pivot && tx < w) {
        sR[ty * stride + tx] = D[(baseT + ty) * V + (baseJ + tx)];
    }
    __syncthreads();

    for (int k = 0; k < pivot; ++k) {
        // Snapshot row k of sR into rowK
        if (ty == k && tx < w) {
            rowK[tx] = sR[k * stride + tx];
        }
        __syncthreads();

        if (ty < pivot && tx < w) {
            int dij = sR[ty * stride + tx];
            int dik = sP[ty * stride + k];
            int dkj = rowK[tx];
            int alt = sat_add_if_valid(dik, dkj);
            if (alt < dij) sR[ty * stride + tx] = alt;
        }
        __syncthreads();
    }

    // Store back
    if (ty < pivot && tx < w) {
        D[(baseT + ty) * V + (baseJ + tx)] = sR[ty * stride + tx];
    }
}

// Phase 2 (col): Update tiles (i, t) for all i != t
__global__ void fw_phase2_col(int* __restrict__ D, int V, int B, int t, int nTiles) {
    extern __shared__ int smem[]; // sP (B*(B+1)) + sC (B*(B+1)) + colK (B)
    const int stride = B + 1;
    int* sP    = smem;
    int* sC    = sP + B * stride;
    int* colK = sC + B * stride;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int iTile = blockIdx.x;
    if (iTile >= nTiles - 1) return;
    const int i = (iTile < t) ? iTile : (iTile + 1);

    const int baseT = t * B;
    const int baseI = i * B;
    const int pivot = (V - baseT) < B ? (V - baseT) : B;
    const int h     = (V - baseI) < B ? (V - baseI) : B;

    // Load pivot tile (pivot x pivot)
    if (ty < pivot && tx < pivot) {
        sP[ty * stride + tx] = D[(baseT + ty) * V + (baseT + tx)];
    }
    // Load col tile (h x pivot)
    if (ty < h && tx < pivot) {
        sC[ty * stride + tx] = D[(baseI + ty) * V + (baseT + tx)];
    }
    __syncthreads();

    for (int k = 0; k < pivot; ++k) {
        // Snapshot column k of sC into colK
        if (tx == k && ty < h) {
            colK[ty] = sC[ty * stride + k];
        }
        __syncthreads();

        if (ty < h && tx < pivot) {
            int dij = sC[ty * stride + tx];
            int dik = colK[ty];
            int dkj = sP[k * stride + tx];
            int alt = sat_add_if_valid(dik, dkj);
            if (alt < dij) sC[ty * stride + tx] = alt;
        }
        __syncthreads();
    }

    // Store back
    if (ty < h && tx < pivot) {
        D[(baseI + ty) * V + (baseT + tx)] = sC[ty * stride + tx];
    }
}

// Phase 3: Update remaining tiles (i, j) for all i != t, j != t
__global__ void fw_phase3_remain(int* __restrict__ D, int V, int B, int t, int nTiles) {
    extern __shared__ int smem[]; // sC (B*(B+1)) + sR (B*(B+1)) + sA (B*(B+1))
    const int stride = B + 1;
    int* sC = smem;
    int* sR = sC + B * stride;
    int* sA = sR + B * stride;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int jTile = blockIdx.x;
    const int iTile = blockIdx.y;
    if (iTile >= nTiles - 1 || jTile >= nTiles - 1) return;
    const int i = (iTile < t) ? iTile : (iTile + 1);
    const int j = (jTile < t) ? jTile : (jTile + 1);

    const int baseT = t * B;
    const int baseI = i * B;
    const int baseJ = j * B;
    const int pivot = (V - baseT) < B ? (V - baseT) : B;
    const int h     = (V - baseI) < B ? (V - baseI) : B;
    const int w     = (V - baseJ) < B ? (V - baseJ) : B;

    // Load tiles: C = (i,t) (h x pivot), R = (t,j) (pivot x w), A = (i,j) (h x w)
    if (ty < h && tx < pivot) {
        sC[ty * stride + tx] = D[(baseI + ty) * V + (baseT + tx)];
    }
    if (ty < pivot && tx < w) {
        sR[ty * stride + tx] = D[(baseT + ty) * V + (baseJ + tx)];
    }
    if (ty < h && tx < w) {
        sA[ty * stride + tx] = D[(baseI + ty) * V + (baseJ + tx)];
    }
    __syncthreads();

    if (ty < h && tx < w) {
        int acc = sA[ty * stride + tx];
#pragma unroll
        for (int k = 0; k < pivot; ++k) {
            int dik = sC[ty * stride + k];
            int dkj = sR[k * stride + tx];
            int alt = sat_add_if_valid(dik, dkj);
            if (alt < acc) acc = alt;
        }
        sA[ty * stride + tx] = acc;
    }

    // Store back
    if (ty < h && tx < w) {
        D[(baseI + ty) * V + (baseJ + tx)] = sA[ty * stride + tx];
    }
}

int main(int argc, char* argv[]){
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file;
    std::string filename = argv[1];
    input_file.open(filename);
    if (!input_file.is_open()) {
        std::cerr << "fileopen error" << filename << std::endl;
        return 1;
    }

    int V, E;
    input_file >> V >> E;

    if (V <= 0) {
        // 无节点，直接退出
        return 0;
    }

    std::vector<int> dist((size_t)V * (size_t)V, INF);
    for (int i = 0; i < V; ++i) dist[(size_t)i * (size_t)V + i] = 0;

    for (int e = 0; e < E; ++e) {
        int u, v, w;
        input_file >> u >> v >> w;
        if (u >= 0 && u < V && v >= 0 && v < V) {
            int &cell = dist[(size_t)u * (size_t)V + v];
            if (w < cell) cell = w; // 取最小权重（稳健防重复）
        }
    }
    input_file.close();

    // 设备端分配
    int *d_D = nullptr;
    HIP_CHECK(hipMalloc(&d_D, sizeof(int) * (size_t)V * (size_t)V));

    HIP_CHECK(hipMemcpy(d_D, dist.data(), sizeof(int) * (size_t)V * (size_t)V, hipMemcpyHostToDevice));

    // 读取分块大小（默认 32）
    int B = 32;
    if (const char* envb = std::getenv("APSP_BLOCK")) {
        int val = std::atoi(envb);
        if (val >= 8) B = val;
    }
    if (B > 32) B = 32; // 限制线程数不超过 1024
    // block 采用 (B,B) 线程组织
    dim3 block(B, B);
    const int nTiles = (V + B - 1) / B;

    // 可选 GPU 计时（仅统计 k 循环内 GPU 时间）
    bool bench = false;
    if (const char* env = std::getenv("APSP_BENCH")) {
        bench = (env[0] != '\0' && env[0] != '0');
    }
    hipEvent_t evStart, evStop;
    if (bench) {
        HIP_CHECK(hipEventCreate(&evStart));
        HIP_CHECK(hipEventCreate(&evStop));
        HIP_CHECK(hipEventRecord(evStart, 0));
    }

    // 三阶段分块 Floyd–Warshall（减少不必要同步）
    for (int t = 0; t < nTiles; ++t) {
        // Phase 1: pivot（必须先完成）
        size_t shmem_p1 = (size_t)B * (size_t)(B + 1) * sizeof(int);
        fw_phase1_pivot<<<1, block, shmem_p1>>>(d_D, V, B, t);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        if (nTiles > 1) {
            // Phase 2: row 和 col 可并行发射
            size_t shmem_p2 = ((size_t)B * (size_t)(B + 1) * 2 + (size_t)B) * sizeof(int);
            fw_phase2_row<<<dim3(nTiles - 1), block, shmem_p2>>>(d_D, V, B, t, nTiles);
            fw_phase2_col<<<dim3(nTiles - 1), block, shmem_p2>>>(d_D, V, B, t, nTiles);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Phase 3: 其余块
            size_t shmem_p3 = (size_t)B * (size_t)(B + 1) * 3 * sizeof(int);
            fw_phase3_remain<<<dim3(nTiles - 1, nTiles - 1), block, shmem_p3>>>(d_D, V, B, t, nTiles);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
        }
    }

    if (bench) {
        HIP_CHECK(hipEventRecord(evStop, 0));
        HIP_CHECK(hipEventSynchronize(evStop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, evStart, evStop));
        std::cerr << "APSP_GPU_MS " << std::fixed << std::setprecision(3) << ms << "\n";
        HIP_CHECK(hipEventDestroy(evStart));
        HIP_CHECK(hipEventDestroy(evStop));
    }

    HIP_CHECK(hipMemcpy(dist.data(), d_D, sizeof(int) * (size_t)V * (size_t)V, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_D));

    // 输出：每行 V 个整数，空格分隔，行末换行
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            std::cout << dist[(size_t)i * (size_t)V + j];
            if (j + 1 < V) std::cout << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}