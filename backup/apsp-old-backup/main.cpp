#include "main.h"
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

    // 调用求解器（GPU 逻辑已迁移到 kernel.hip 中）
    solve(dist.data(), V);

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