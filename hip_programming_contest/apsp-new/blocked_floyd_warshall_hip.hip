#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <iostream>
#define INF INT_MAX / 2

// Floyd-Warshall分块算法的三个阶段的kernel函数声明
// 这些是__global__函数，只能在.hip文件中调用

extern "C" {
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
}

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(e) << "\n"; \
        std::exit(1); \
    } \
} while(0)



__forceinline__
__device__ int get(int *graph, int flat_index, int n, int oversized_n) {
	int row_index = flat_index / oversized_n;
	int col_index = flat_index % oversized_n;
	if (row_index >= n || col_index >= n) {
		return INF;
	} else {
		return graph[row_index * n + col_index];
	}
}

__forceinline__
__device__ void update(int *graph, int flat_index, int n, int oversized_n, int value) {
	int row_index = flat_index / oversized_n;
	int col_index = flat_index % oversized_n;
	if (row_index < n && col_index < n) {
		graph[row_index * n + col_index] = value;
	}
}

__forceinline__
__device__ void block_calc(int* C, int* A, int* B, int bj, int bi, int b, int stride) {
  #pragma unroll
  for (int k = 0; k < b; k++) {
    int sum = A[bi*stride + k] + B[k*stride + bj];
    if (C[bi*stride + bj] > sum) {
      C[bi*stride + bj] = sum;
    }
    __syncthreads();
  }
}

extern "C" __global__ void floyd_warshall_block_kernel_phase1(int n, int oversized_n, int k, int* graph, int b) {
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  extern __shared__ int phase1_shared[];
  const int stride = b + 1; // Add padding to avoid bank conflicts
  int *C = phase1_shared;

  __syncthreads();

  // Transfer to temp shared arrays
  int i = k*b*oversized_n + k*b + bi*oversized_n + bj;
  C[bi*stride + bj] = get(graph, i, n, oversized_n);

  __syncthreads();

  block_calc(C, C, C, bi, bj, b, stride);

  __syncthreads();

  // Transfer back to graph
  update(graph, i, n, oversized_n, C[bi*stride+bj]);
}

extern "C" __global__ void floyd_warshall_block_kernel_phase2_row(int n, int oversized_n, int k, int* graph, int b) {
  const unsigned int i = blockIdx.x;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k) return;

  extern __shared__ int phase2_shared[];
  const int stride = b + 1; // Add padding to avoid bank conflicts
  int *A = phase2_shared;
  int *B = (int*)&A[b * stride];
  int *C = (int*)&B[b * stride];

  __syncthreads();
  int i1 = i*b*oversized_n + k*b + bi*oversized_n + bj;
  C[bi*stride + bj] = get(graph, i1, n, oversized_n);
  int i2 = k*b*oversized_n + k*b + bi*oversized_n + bj;
  B[bi*stride + bj] = get(graph, i2, n, oversized_n);

  __syncthreads();

  block_calc(C, C, B, bi, bj, b, stride);

  __syncthreads();

  update(graph, i1, n, oversized_n, C[bi*stride+bj]);
}

extern "C" __global__ void floyd_warshall_block_kernel_phase2_col(int n, int oversized_n, int k, int* graph, int b) {
  const unsigned int j = blockIdx.x;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (j == k) return;

  extern __shared__ int phase2_shared[];
  const int stride = b + 1; // Add padding to avoid bank conflicts
  int *A = phase2_shared;
  int *B = (int*)&A[b * stride];
  int *C = (int*)&B[b * stride];

  __syncthreads();
  int i1 = k*b*oversized_n + j*b + bi*oversized_n + bj;
  C[bi*stride + bj] = get(graph, i1, n, oversized_n);
  int i2 = k*b*oversized_n + k*b + bi*oversized_n + bj;
  A[bi*stride + bj] = get(graph, i2, n, oversized_n);

  __syncthreads();

  block_calc(C, A, C, bi, bj, b, stride);

  __syncthreads();

  update(graph, i1, n, oversized_n, C[bi*stride+bj]);
}

extern "C" __global__ void floyd_warshall_block_kernel_phase3(int n, int oversized_n, int k, int* graph, int b) {
  const unsigned int j = blockIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k && j == k) return;

  extern __shared__ int phase3_shared[];
  const int stride = b + 1; // Add padding to avoid bank conflicts
  int *A = phase3_shared;
  int *B = (int*)&A[b * stride];
  int *C = (int*)&B[b * stride];

  __syncthreads();

  int i1 = i*b*oversized_n + j*b + bi*oversized_n + bj;
  C[bi*stride + bj] = get(graph, i1, n, oversized_n);
  int i2 = i*b*oversized_n + k*b + bi*oversized_n + bj;
  A[bi*stride + bj] = get(graph, i2, n, oversized_n);
  int i3 = k*b*oversized_n + j*b + bi*oversized_n + bj;
  B[bi*stride + bj] = get(graph, i3, n, oversized_n);

  __syncthreads();

  block_calc(C, A, B, bi, bj, b, stride);

  __syncthreads();

  update(graph, i1, n, oversized_n, C[bi*stride+bj]);
}

