#include "main.h"
#include <cstdlib>

// 常量：不可达的距离值
static constexpr int INF = 1073741823; // 2^30 - 1

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(e) << "\n"; \
        std::exit(1); \
    } \
} while(0)

__global__ void extractRowColKernel(const int* __restrict__ D, int V, int k, int* __restrict__ rowK, int* __restrict__ colK) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        rowK[idx] = D[k * V + idx];
        colK[idx] = D[idx * V + k];
    }
}

__global__ void fwUpdateKernel(int* __restrict__ D, const int* __restrict__ rowK, const int* __restrict__ colK, int V) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= V || j >= V) return;

    int dij = D[i * V + j];
    int dik = colK[i];
    int dkj = rowK[j];

    // 若任一为不可达，则替代路径不可用
    if (dik >= INF || dkj >= INF) return;

    // 防溢出加法，并限制到 INF
    long long alt64 = (long long)dik + (long long)dkj;
    if (alt64 > (long long)INF) alt64 = (long long)INF;
    int alt = (int)alt64;

    if (alt < dij) {
        D[i * V + j] = alt;
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
    int *d_rowK = nullptr;
    int *d_colK = nullptr;
    HIP_CHECK(hipMalloc(&d_D, sizeof(int) * (size_t)V * (size_t)V));
    HIP_CHECK(hipMalloc(&d_rowK, sizeof(int) * (size_t)V));
    HIP_CHECK(hipMalloc(&d_colK, sizeof(int) * (size_t)V));

    HIP_CHECK(hipMemcpy(d_D, dist.data(), sizeof(int) * (size_t)V * (size_t)V, hipMemcpyHostToDevice));

    const int tpb1D = 256;
    dim3 block1D(tpb1D);
    dim3 grid1D((V + tpb1D - 1) / tpb1D);

    const int bx = 16, by = 16;
    dim3 block2D(bx, by);
    dim3 grid2D((V + bx - 1) / bx, (V + by - 1) / by);

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

    for (int k = 0; k < V; ++k) {
        extractRowColKernel<<<grid1D, block1D>>>(d_D, V, k, d_rowK, d_colK);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        fwUpdateKernel<<<grid2D, block2D>>>(d_D, d_rowK, d_colK, V);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
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
    HIP_CHECK(hipFree(d_rowK));
    HIP_CHECK(hipFree(d_colK));

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