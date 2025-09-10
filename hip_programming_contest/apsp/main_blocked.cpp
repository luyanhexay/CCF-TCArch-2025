#include "blocked_floyd_warshall.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

static constexpr int INF = 1073741823; // 2^30 - 1

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream input_file;
    std::string filename = argv[1];
    input_file.open(filename);
    if (!input_file.is_open())
    {
        std::cerr << "fileopen error: " << filename << std::endl;
        return 1;
    }

    int V, E;
    input_file >> V >> E;

    if (V <= 0)
    {
        // 无节点，直接退出
        return 0;
    }

    // 初始化距离矩阵
    std::vector<int> dist((size_t)V * (size_t)V, INF);
    for (int i = 0; i < V; ++i)
    {
        dist[(size_t)i * (size_t)V + i] = 0;
    }

    // 读取边信息
    for (int e = 0; e < E; ++e)
    {
        int u, v, w;
        input_file >> u >> v >> w;
        if (u >= 0 && u < V && v >= 0 && v < V)
        {
            int &cell = dist[(size_t)u * (size_t)V + v];
            if (w < cell)
                cell = w; // 取最小权重（稳健防重复）
        }
    }
    input_file.close();

    // 选择块大小 - 可以根据V的大小动态调整
    int block_size = 32; // 默认块大小
    if (V <= 1000)
    {
        block_size = 16;
    }
    else if (V <= 5000)
    {
        block_size = 32;
    }
    else if (V <= 20000)
    {
        block_size = 64;
    }
    else
    {
        block_size = 128;
    }

    // 确保块大小不超过V
    if (block_size > V)
    {
        block_size = V;
    }

    std::cout << "Using block size: " << block_size << std::endl;

    // 调用分块Floyd-Warshall算法
    t_error err;
    auto start = std::chrono::high_resolution_clock::now();

    int result = blocked_floyd_warshall(dist.data(), V, block_size, dist.data(), &err);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result != 0)
    {
        std::cerr << "Error in blocked_floyd_warshall: " << err.err_msg << std::endl;
        return 1;
    }

    std::cout << "Computation time: " << duration.count() << " ms" << std::endl;

    // 输出结果：每行V个整数，空格分隔，行末换行
    for (int i = 0; i < V; ++i)
    {
        for (int j = 0; j < V; ++j)
        {
            std::cout << dist[(size_t)i * (size_t)V + j];
            if (j + 1 < V)
                std::cout << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}
