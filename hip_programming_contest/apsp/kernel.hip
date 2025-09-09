#include "main.h"
#include <cstdlib>
#include <algorithm>

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
    // Additional shared buffers to snapshot pivot row/col for correctness
    const int stride = B + 1; // keep padding consistent with usage below
    int* rowK = sP + B * stride;
    int* colK = rowK + B;

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int base = t * B;
    const int tile = (V - base) < B ? (V - base) : B;

    // padding to mitigate bank conflicts
    // Load pivot tile into shared（标量）
    if (ty < tile && tx < tile) {
        sP[ty * stride + tx] = D[(base + ty) * V + (base + tx)];
    }
    __syncthreads();

    // In-place FW within the tile with row/col snapshot to avoid hazards
    for (int k = 0; k < tile; ++k) {
        // Snapshot pivot row k and column k before updates
        if (ty == k && tx < tile) {
            rowK[tx] = sP[k * stride + tx];
        }
        if (tx == k && ty < tile) {
            colK[ty] = sP[ty * stride + k];
        }
        __syncthreads();

        if (ty < tile && tx < tile) {
            int dij = sP[ty * stride + tx];
            int dik = colK[ty];
            int dkj = rowK[tx];
            int alt = sat_add_if_valid(dik, dkj);
            if (alt < dij) sP[ty * stride + tx] = alt;
        }
        __syncthreads();
    }

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

    // 2x2 cooperative loads into shared memory for C (h x pivot), R (pivot x w), A (h x w)
    const int row0 = 2 * ty;
    const int col0 = 2 * tx;
    // Load sC: rows in h-dim, cols in pivot-dim
    for (int ry = 0; ry < 2; ++ry) {
        int r = row0 + ry;
        if (r < h) {
            for (int rx = 0; rx < 2; ++rx) {
                int c = col0 + rx; // reuse tx across pivot-dim
                if (c < pivot) {
                    sC[r * stride + c] = D[(baseI + r) * V + (baseT + c)];
                }
            }
        }
    }
    // Load sR: rows in pivot-dim, cols in w-dim
    for (int ry = 0; ry < 2; ++ry) {
        int r = row0 + ry; // reuse ty across pivot-dim
        if (r < pivot) {
            for (int rx = 0; rx < 2; ++rx) {
                int c = col0 + rx;
                if (c < w) {
                    sR[r * stride + c] = D[(baseT + r) * V + (baseJ + c)];
                }
            }
        }
    }
    // Load sA: rows in h-dim, cols in w-dim
    for (int ry = 0; ry < 2; ++ry) {
        int r = row0 + ry;
        if (r < h) {
            for (int rx = 0; rx < 2; ++rx) {
                int c = col0 + rx;
                if (c < w) {
                    sA[r * stride + c] = D[(baseI + r) * V + (baseJ + c)];
                }
            }
        }
    }
    __syncthreads();

    // Each thread computes a 2x2 micro-tile
    {
        const int r0 = row0;
        const int r1 = row0 + 1;
        const int c0 = col0;
        const int c1 = col0 + 1;

        const bool v00 = (r0 < h) && (c0 < w);
        const bool v01 = (r0 < h) && (c1 < w);
        const bool v10 = (r1 < h) && (c0 < w);
        const bool v11 = (r1 < h) && (c1 < w);

        int acc00 = v00 ? sA[r0 * stride + c0] : INF;
        int acc01 = v01 ? sA[r0 * stride + c1] : INF;
        int acc10 = v10 ? sA[r1 * stride + c0] : INF;
        int acc11 = v11 ? sA[r1 * stride + c1] : INF;

        #pragma unroll
        for (int k = 0; k < pivot; ++k) {
            int c_r0k = (r0 < h) ? sC[r0 * stride + k] : INF;
            int c_r1k = (r1 < h) ? sC[r1 * stride + k] : INF;
            int r_kc0 = (c0 < w) ? sR[k * stride + c0] : INF;
            int r_kc1 = (c1 < w) ? sR[k * stride + c1] : INF;

            if (v00) {
                int alt = sat_add_if_valid(c_r0k, r_kc0);
                if (alt < acc00) acc00 = alt;
            }
            if (v01) {
                int alt = sat_add_if_valid(c_r0k, r_kc1);
                if (alt < acc01) acc01 = alt;
            }
            if (v10) {
                int alt = sat_add_if_valid(c_r1k, r_kc0);
                if (alt < acc10) acc10 = alt;
            }
            if (v11) {
                int alt = sat_add_if_valid(c_r1k, r_kc1);
                if (alt < acc11) acc11 = alt;
            }
        }

        if (v00) sA[r0 * stride + c0] = acc00;
        if (v01) sA[r0 * stride + c1] = acc01;
        if (v10) sA[r1 * stride + c0] = acc10;
        if (v11) sA[r1 * stride + c1] = acc11;
    }

    // Store back 2x2 micro-tile
    for (int ry = 0; ry < 2; ++ry) {
        int r = row0 + ry;
        if (r < h) {
            for (int rx = 0; rx < 2; ++rx) {
                int c = col0 + rx;
                if (c < w) {
                    D[(baseI + r) * V + (baseJ + c)] = sA[r * stride + c];
                }
            }
        }
    }
}

extern "C" void solve(int* dist, int V) {
    if (V <= 0) return;

    // 设备端分配
    int *d_D = nullptr;
    HIP_CHECK(hipMalloc(&d_D, sizeof(int) * (size_t)V * (size_t)V));
    HIP_CHECK(hipMemcpy(d_D, dist, sizeof(int) * (size_t)V * (size_t)V, hipMemcpyHostToDevice));

    // 读取分块大小（默认 16，且约束为偶数且在 [8,32]）
    int B = 16;
    if (const char* envb = std::getenv("APSP_BLOCK")) {
        int val = std::atoi(envb);
        if (val >= 8) B = val;
    }
    if (B > 32) B = 32; // 限制线程数不超过 1024
    if (B < 8) B = 8;
    if (B & 1) B -= 1;  // 确保为偶数，便于 2x2 微瓦片

    // block：Phase1/2 采用 (B,B)；Phase3 采用 (B/2,B/2) 以做 2x2 微瓦片
    dim3 block12(B, B);
    dim3 block3(B / 2, B / 2);
    const int nTiles = (V + B - 1) / B;

    // 三阶段分块 Floyd–Warshall
    for (int t = 0; t < nTiles; ++t) {
        // Phase 1: pivot（必须先完成）
        size_t shmem_p1 = ((size_t)B * (size_t)(B + 1) + (size_t)2 * (size_t)B) * sizeof(int);
        fw_phase1_pivot<<<1, block12, shmem_p1>>>(d_D, V, B, t);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        if (nTiles > 1) {
            // Phase 2: row 和 col 可并行发射
            size_t shmem_p2 = ((size_t)B * (size_t)(B + 1) * 2 + (size_t)B) * sizeof(int);
            fw_phase2_row<<<dim3(nTiles - 1), block12, shmem_p2>>>(d_D, V, B, t, nTiles);
            fw_phase2_col<<<dim3(nTiles - 1), block12, shmem_p2>>>(d_D, V, B, t, nTiles);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Phase 3: 其余块
            size_t shmem_p3 = (size_t)B * (size_t)(B + 1) * 3 * sizeof(int);
            fw_phase3_remain<<<dim3(nTiles - 1, nTiles - 1), block3, shmem_p3>>>(d_D, V, B, t, nTiles);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
        }
    }

    HIP_CHECK(hipMemcpy(dist, d_D, sizeof(int) * (size_t)V * (size_t)V, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_D));
}


