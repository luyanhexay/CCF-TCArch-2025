#include "main.h"
#include "apsp_solver.h"

// 手动实现min函数，使用不同的名称避免与HIP内置函数冲突
inline int my_min(int a, int b) {
    return (a < b) ? a : b;
}

extern "C" void solve(int* dist, int V) {
    if (V <= 0) return;
    
    // 使用新的APSP_Solver架构
    APSP_Solver solver;
    
    // 初始化求解器（一次性数据传输）
    // 使用合适的块大小，不能超过图的节点数
    int block_size = my_min(16, V);
    int result = solver.init(dist, V, block_size);
    if (result != 0) {
        std::cerr << "Error initializing APSP_Solver: " << solver.get_last_error() << std::endl;
        return;
    }
    
    // 执行GPU计算（纯设备计算，无数据传输）
    result = solver.solve();
    if (result != 0) {
        std::cerr << "Error in APSP_Solver::solve(): " << solver.get_last_error() << std::endl;
        return;
    }
    
    // 获取结果（一次性数据传输）
    result = solver.get_result(dist);
    if (result != 0) {
        std::cerr << "Error getting result from APSP_Solver: " << solver.get_last_error() << std::endl;
        return;
    }
}
