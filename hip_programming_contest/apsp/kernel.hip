#include "main.h"
#include "apsp_solver.h"
#include <algorithm>

extern "C" void solve(int* dist, int V) {
    if (V <= 0) return;
    
    // 使用新的APSP_Solver架构
    APSP_Solver solver;
    
    // 初始化求解器（一次性数据传输）
    // 使用合适的块大小，不能超过图的节点数
    int block_size = std::min(16, V);
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
