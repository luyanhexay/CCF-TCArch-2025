#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <iostream>
extern "C" {
    #include "blocked_floyd_warshall.h"
}
#define INF INT_MAX / 2

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(e) << "\n"; \
        std::exit(1); \
    } \
} while(0)

__forceinline__
__host__ int check_hip_error(t_error* err) {
	hipError_t errCode = hipGetLastError();
	if (errCode != hipSuccess) {
		err->err_code = errCode;
		snprintf(err->err_msg, sizeof(err->err_msg), "%s", hipGetErrorString(errCode));
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}


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

__global__ void floyd_warshall_block_kernel_phase1(int n, int oversized_n, int k, int* graph, int b) {
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

__global__ void floyd_warshall_block_kernel_phase2_row(int n, int oversized_n, int k, int* graph, int b) {
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

__global__ void floyd_warshall_block_kernel_phase2_col(int n, int oversized_n, int k, int* graph, int b) {
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

__global__ void floyd_warshall_block_kernel_phase3(int n, int oversized_n, int k, int* graph, int b) {
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

extern "C"
__host__ int blocked_floyd_warshall(int* graph, int n, int b, int* apsp, t_error* err) {

	if (n <= 0) {
		err->err_code = WRONG_NUM_OF_NODES_ERR;
		char err_msg[100] = {
				"The number of nodes should be > 0"};
		snprintf(err->err_msg, sizeof(err->err_msg), "%s", err_msg);
		return EXIT_FAILURE;
	}

	if (b <= 0 || b > n) {
		err->err_code = WRONG_BLOCK_SIZE_ERR;
		char err_msg[100] = {
				"The number of blocks b should be: 0 < b <= n, where n is the number of nodes"};
		snprintf(err->err_msg, sizeof(err->err_msg), "%s", err_msg);
		return EXIT_FAILURE;
	}

	int oversized_n;
	int block_remainder = n % b;
	if (block_remainder == 0) {
		oversized_n = n;
	} else {
		oversized_n = n + b - block_remainder;
	}
	const size_t size = n * n * sizeof(int);
	int* device_graph;

	hipError_t err1 = hipMalloc(&device_graph, size);
	if (err1 != hipSuccess) {
		err->err_code = err1;
		snprintf(err->err_msg, sizeof(err->err_msg), "hipMalloc failed: %s", hipGetErrorString(err1));
		return EXIT_FAILURE;
	}
	hipError_t err2 = hipMemcpy(device_graph, graph,  size, hipMemcpyHostToDevice);
	if (err2 != hipSuccess) {
		err->err_code = err2;
		snprintf(err->err_msg, sizeof(err->err_msg), "hipMemcpy H2D failed: %s", hipGetErrorString(err2));
		hipFree(device_graph);
		return EXIT_FAILURE;
	}

	const int blocks = oversized_n / b;
	dim3 block_dim(b, b, 1);
	dim3 phase3_grid(blocks, blocks, 1);

	const int stride = b + 1; // Add padding to avoid bank conflicts
	const size_t block_size = b * stride * sizeof(int);

	int k;
	for (k = 0; k < blocks; k++) {
		floyd_warshall_block_kernel_phase1<<<1, block_dim, block_size>>>(n, oversized_n, k, device_graph, b);
		floyd_warshall_block_kernel_phase2_row<<<blocks, block_dim, block_size*3>>>(n, oversized_n, k, device_graph, b);
		floyd_warshall_block_kernel_phase2_col<<<blocks, block_dim, block_size*3>>>(n, oversized_n, k, device_graph, b);
		floyd_warshall_block_kernel_phase3<<<phase3_grid, block_dim, block_size*3>>>(n, oversized_n, k, device_graph, b);
	}

	hipError_t err3 = hipMemcpy(apsp, device_graph, size, hipMemcpyDeviceToHost);
	if (err3 != hipSuccess) {
		err->err_code = err3;
		snprintf(err->err_msg, sizeof(err->err_msg), "hipMemcpy D2H failed: %s", hipGetErrorString(err3));
		hipFree(device_graph);
		return EXIT_FAILURE;
	}
	hipError_t err4 = hipFree(device_graph);
	if (err4 != hipSuccess) {
		err->err_code = err4;
		snprintf(err->err_msg, sizeof(err->err_msg), "hipFree failed: %s", hipGetErrorString(err4));
		return EXIT_FAILURE;
	}

	return check_hip_error(err);
}
