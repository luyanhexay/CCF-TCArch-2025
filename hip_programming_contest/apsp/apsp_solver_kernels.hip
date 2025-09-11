#include "apsp_solver.h"
#include "kernels.h"

// 这个文件包含APSP_Solver中需要在GPU上执行的kernel调用
// 因为kernel调用必须在.hip文件中进行

extern "C" int apsp_solver_solve_gpu(int V, int oversized_n, int block_size, int* d_graph) {
    const int blocks = oversized_n / block_size;
    dim3 block_dim(block_size, block_size, 1);
    dim3 phase3_grid(blocks, blocks, 1);
    
    const int stride = block_size + 1; // Add padding to avoid bank conflicts
    const size_t block_size_bytes = block_size * stride * sizeof(int);
    
    // 直接在GPU内存上执行Floyd-Warshall算法，无数据传输
    // 这是真正的纯GPU计算实现
    
    // 执行Floyd-Warshall算法的三个阶段
    for (int k = 0; k < blocks; k++) {
        // Phase 1: 对角线块更新
        floyd_warshall_block_kernel_phase1<<<1, block_dim, block_size_bytes>>>(
            V, oversized_n, k, d_graph, block_size);
        
        // Phase 2: 行和列更新
        floyd_warshall_block_kernel_phase2_row<<<blocks, block_dim, block_size_bytes*3>>>(
            V, oversized_n, k, d_graph, block_size);
        floyd_warshall_block_kernel_phase2_col<<<blocks, block_dim, block_size_bytes*3>>>(
            V, oversized_n, k, d_graph, block_size);
        
        // Phase 3: 剩余块更新
        floyd_warshall_block_kernel_phase3<<<phase3_grid, block_dim, block_size_bytes*3>>>(
            V, oversized_n, k, d_graph, block_size);
    }
    
    // 不需要显式同步，HIP会在函数返回前自动同步
    return 0;
}
