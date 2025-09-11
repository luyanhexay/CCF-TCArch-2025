#include "apsp_solver.h"
#include "kernels.h"
#include <cstring>

// HIP错误检查宏
#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        set_error(e, hipGetErrorString(e)); \
        return -1; \
    } \
} while(0)

APSP_Solver::APSP_Solver() 
    : d_graph(nullptr)
    , allocated_size(0)
    , current_V(0)
    , current_block_size(16)
    , h_graph_pinned(nullptr)
    , h_result_pinned(nullptr)
    , pinned_size(0)
    , using_memory_pool(false)
    , initialized(false)
    , graph_loaded(false)
{
    last_error.code = 0;
    last_error.message[0] = '\0';
}

APSP_Solver::~APSP_Solver() {
    cleanup_memory();
}

void APSP_Solver::set_error(int code, const char* message) {
    last_error.code = code;
    strncpy(last_error.message, message, sizeof(last_error.message) - 1);
    last_error.message[sizeof(last_error.message) - 1] = '\0';
}

int APSP_Solver::allocate_gpu_memory(int V) {
    size_t required_size = (size_t)V * V * sizeof(int);
    
    // 如果已分配的内存足够，直接使用
    if (allocated_size >= required_size && d_graph != nullptr) {
        return 0;
    }
    
    // 释放旧内存
    if (d_graph != nullptr) {
        if (using_memory_pool) {
            g_memory_pool.release_gpu_memory(d_graph);
        } else {
            HIP_CHECK(hipFree(d_graph));
        }
        d_graph = nullptr;
    }
    
    // 分配新内存
    if (using_memory_pool) {
        d_graph = (int*)g_memory_pool.get_gpu_memory(required_size);
        if (d_graph == nullptr) {
            set_error(-1, "Failed to get GPU memory from pool");
            return -1;
        }
    } else {
        HIP_CHECK(hipMalloc(&d_graph, required_size));
    }
    allocated_size = required_size;
    
    return 0;
}

int APSP_Solver::allocate_pinned_memory(int V) {
    size_t required_size = (size_t)V * V * sizeof(int);
    
    // 如果已分配的页锁定内存足够，直接使用
    if (pinned_size >= required_size && h_graph_pinned != nullptr && h_result_pinned != nullptr) {
        return 0;
    }
    
    // 释放旧内存
    if (h_graph_pinned != nullptr) {
        if (using_memory_pool) {
            g_memory_pool.release_pinned_memory(h_graph_pinned);
        } else {
            HIP_CHECK(hipHostFree(h_graph_pinned));
        }
        h_graph_pinned = nullptr;
    }
    if (h_result_pinned != nullptr) {
        if (using_memory_pool) {
            g_memory_pool.release_pinned_memory(h_result_pinned);
        } else {
            HIP_CHECK(hipHostFree(h_result_pinned));
        }
        h_result_pinned = nullptr;
    }
    
    // 分配新的页锁定内存
    if (using_memory_pool) {
        h_graph_pinned = (int*)g_memory_pool.get_pinned_memory(required_size);
        h_result_pinned = (int*)g_memory_pool.get_pinned_memory(required_size);
        
        if (h_graph_pinned == nullptr || h_result_pinned == nullptr) {
            set_error(-1, "Failed to get pinned memory from pool");
            return -1;
        }
    } else {
        HIP_CHECK(hipHostMalloc(&h_graph_pinned, required_size));
        HIP_CHECK(hipHostMalloc(&h_result_pinned, required_size));
    }
    pinned_size = required_size;
    
    return 0;
}

void APSP_Solver::cleanup_memory() {
    if (d_graph != nullptr) {
        if (using_memory_pool) {
            g_memory_pool.release_gpu_memory(d_graph);
        } else {
            hipFree(d_graph);
        }
        d_graph = nullptr;
    }
    
    if (h_graph_pinned != nullptr) {
        if (using_memory_pool) {
            g_memory_pool.release_pinned_memory(h_graph_pinned);
        } else {
            hipHostFree(h_graph_pinned);
        }
        h_graph_pinned = nullptr;
    }
    
    if (h_result_pinned != nullptr) {
        if (using_memory_pool) {
            g_memory_pool.release_pinned_memory(h_result_pinned);
        } else {
            hipHostFree(h_result_pinned);
        }
        h_result_pinned = nullptr;
    }
    
    allocated_size = 0;
    pinned_size = 0;
    initialized = false;
    graph_loaded = false;
}

int APSP_Solver::init(const int* graph, int V, int block_size) {
    if (V <= 0) {
        set_error(-1, "Invalid vertex count: V must be > 0");
        return -1;
    }
    
    if (block_size <= 0 || block_size > V) {
        set_error(-1, "Invalid block size: must be 0 < block_size <= V");
        return -1;
    }
    
    // 重置状态
    reset();
    
    // 分配GPU内存
    if (allocate_gpu_memory(V) != 0) {
        return -1;
    }
    
    // 分配页锁定主机内存
    if (allocate_pinned_memory(V) != 0) {
        return -1;
    }
    
    // 将输入图数据拷贝到页锁定内存
    memcpy(h_graph_pinned, graph, (size_t)V * V * sizeof(int));
    
    // 将图数据从页锁定内存传输到GPU
    HIP_CHECK(hipMemcpy(d_graph, h_graph_pinned, (size_t)V * V * sizeof(int), hipMemcpyHostToDevice));
    
    // 更新状态
    current_V = V;
    current_block_size = block_size;
    initialized = true;
    graph_loaded = true;
    
    return 0;
}

int APSP_Solver::solve() {
    if (!initialized || !graph_loaded) {
        set_error(-1, "Solver not initialized or graph not loaded");
        return -1;
    }
    
    // 使用现有的blocked_floyd_warshall函数，但传入GPU内存指针
    // 注意：我们需要修改blocked_floyd_warshall函数以支持GPU到GPU的操作
    // 暂时使用一个简化的实现
    
    // 计算oversized_n
    int oversized_n;
    int block_remainder = current_V % current_block_size;
    if (block_remainder == 0) {
        oversized_n = current_V;
    } else {
        oversized_n = current_V + current_block_size - block_remainder;
    }
    
    const int blocks = oversized_n / current_block_size;
    dim3 block_dim(current_block_size, current_block_size, 1);
    dim3 phase3_grid(blocks, blocks, 1);
    
    const int stride = current_block_size + 1; // Add padding to avoid bank conflicts
    const size_t block_size_bytes = current_block_size * stride * sizeof(int);
    
    // 调用GPU kernel函数执行纯GPU计算
    int result = apsp_solver_solve_gpu(current_V, oversized_n, current_block_size, d_graph);
    
    if (result != 0) {
        set_error(-1, "GPU kernel execution failed");
        return -1;
    }
    
    return 0;
}

int APSP_Solver::get_result(int* result) {
    if (!initialized || !graph_loaded) {
        set_error(-1, "Solver not initialized or graph not loaded");
        return -1;
    }
    
    // 将结果从GPU传输到页锁定内存
    HIP_CHECK(hipMemcpy(h_result_pinned, d_graph, (size_t)current_V * current_V * sizeof(int), hipMemcpyDeviceToHost));
    
    // 将结果从页锁定内存拷贝到用户缓冲区
    memcpy(result, h_result_pinned, (size_t)current_V * current_V * sizeof(int));
    
    return 0;
}

void APSP_Solver::reset() {
    cleanup_memory();
    current_V = 0;
    current_block_size = 16;
    last_error.code = 0;
    last_error.message[0] = '\0';
}
