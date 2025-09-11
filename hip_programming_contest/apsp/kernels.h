#ifndef KERNELS_H
#define KERNELS_H

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Floyd-Warshall分块算法的三个阶段的kernel函数声明
// 这些是__global__函数，只能在.hip文件中调用

/**
 * Phase 1: 对角线块更新
 * 计算最短路径，考虑对角线块内的顶点作为中间点
 */
__global__ void floyd_warshall_block_kernel_phase1(int n, int oversized_n, int k, int* graph, int b);

/**
 * Phase 2: 行更新
 * 使用新计算的对角线块距离更新同一行的块
 */
__global__ void floyd_warshall_block_kernel_phase2_row(int n, int oversized_n, int k, int* graph, int b);

/**
 * Phase 2: 列更新
 * 使用新计算的对角线块距离更新同一列的块
 */
__global__ void floyd_warshall_block_kernel_phase2_col(int n, int oversized_n, int k, int* graph, int b);

/**
 * Phase 3: 剩余块更新
 * 对于所有不在对角线、行、列上的块，更新距离
 */
__global__ void floyd_warshall_block_kernel_phase3(int n, int oversized_n, int k, int* graph, int b);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H
