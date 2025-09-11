#include "memory_pool.h"
#include <iostream>
#include <algorithm>

// HIP错误检查宏
#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        std::cerr << "MemoryPool HIP error: " << hipGetErrorString(e) << std::endl; \
        return nullptr; \
    } \
} while(0)

MemoryPool::MemoryPool() {
    // 完全禁用预分配，改为纯按需分配
    // preallocate_small_sizes(); // 注释掉预分配
}

MemoryPool::~MemoryPool() {
    cleanup_all();
}

void MemoryPool::preallocate_small_sizes() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // 智能预分配策略：预分配常用大小，包括6400×6400图
    const int SMART_SIZES_COUNT = 7; // 包括6400×6400图
    
    // 预分配GPU内存块
    for (size_t i = 0; i < SMART_SIZES_COUNT; i++) {
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, COMMON_SIZES[i]);
        if (err == hipSuccess && ptr != nullptr) {
            gpu_blocks.push_back({ptr, COMMON_SIZES[i], false});
        }
    }
    
    // 预分配页锁定内存块
    for (size_t i = 0; i < SMART_SIZES_COUNT; i++) {
        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, COMMON_SIZES[i]);
        if (err == hipSuccess && ptr != nullptr) {
            pinned_blocks.push_back({ptr, COMMON_SIZES[i], false});
        }
    }
}

void MemoryPool::preallocate_common_sizes() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // 预分配常用的GPU内存块
    for (size_t i = 0; i < NUM_COMMON_SIZES; i++) {
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, COMMON_SIZES[i]);
        if (err == hipSuccess && ptr != nullptr) {
            gpu_blocks.push_back({ptr, COMMON_SIZES[i], false});
        }
    }
    
    // 预分配常用的页锁定内存块
    for (size_t i = 0; i < NUM_COMMON_SIZES; i++) {
        void* ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, COMMON_SIZES[i]);
        if (err == hipSuccess && ptr != nullptr) {
            pinned_blocks.push_back({ptr, COMMON_SIZES[i], false});
        }
    }
    
    // 静默初始化，不输出调试信息
    // std::cout << "MemoryPool: Preallocated " << gpu_blocks.size() 
    //           << " GPU blocks and " << pinned_blocks.size() 
    //           << " pinned blocks" << std::endl;
}

void* MemoryPool::find_or_allocate_gpu_block(size_t size) {
    // 首先尝试找到合适大小的未使用块
    for (auto& block : gpu_blocks) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            return block.ptr;
        }
    }
    
    // 如果没有找到合适的块，使用延迟分配策略
    return allocate_large_block(size);
}

void* MemoryPool::allocate_large_block(size_t size) {
    // 对于大内存块，使用优化的分配策略
    void* ptr = nullptr;
    
    // 直接分配，不加入池管理（避免池管理开销）
    hipError_t err = hipMalloc(&ptr, size);
    
    if (err == hipSuccess && ptr != nullptr) {
        // 不加入池管理，直接返回
        return ptr;
    }
    
    return nullptr;
}

void* MemoryPool::find_or_allocate_pinned_block(size_t size) {
    // 首先尝试找到合适大小的未使用块
    for (auto& block : pinned_blocks) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            return block.ptr;
        }
    }
    
    // 如果没有找到合适的块，分配新的
    void* ptr = nullptr;
    HIP_CHECK(hipHostMalloc(&ptr, size));
    
    if (ptr != nullptr) {
        pinned_blocks.push_back({ptr, size, true});
    }
    
    return ptr;
}

void* MemoryPool::get_gpu_memory(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return find_or_allocate_gpu_block(size);
}

void* MemoryPool::get_pinned_memory(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return find_or_allocate_pinned_block(size);
}

void MemoryPool::release_gpu_memory(void* ptr) {
    if (ptr == nullptr) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    for (auto& block : gpu_blocks) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
    
    // 如果没找到，说明不是从池中分配的，直接释放
    hipFree(ptr);
}

void MemoryPool::release_pinned_memory(void* ptr) {
    if (ptr == nullptr) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    for (auto& block : pinned_blocks) {
        if (block.ptr == ptr) {
            block.in_use = false;
            return;
        }
    }
    
    // 如果没找到，说明不是从池中分配的，直接释放
    hipHostFree(ptr);
}

void MemoryPool::get_stats(size_t& gpu_blocks_count, size_t& pinned_blocks_count, 
                           size_t& gpu_total_size, size_t& pinned_total_size) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    gpu_blocks_count = gpu_blocks.size();
    pinned_blocks_count = pinned_blocks.size();
    
    gpu_total_size = 0;
    pinned_total_size = 0;
    
    for (const auto& block : gpu_blocks) {
        gpu_total_size += block.size;
    }
    
    for (const auto& block : pinned_blocks) {
        pinned_total_size += block.size;
    }
}

void MemoryPool::cleanup_all() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    // 释放所有GPU内存
    for (const auto& block : gpu_blocks) {
        if (block.ptr != nullptr) {
            hipFree(block.ptr);
        }
    }
    gpu_blocks.clear();
    
    // 释放所有页锁定内存
    for (const auto& block : pinned_blocks) {
        if (block.ptr != nullptr) {
            hipHostFree(block.ptr);
        }
    }
    pinned_blocks.clear();
}

// 全局内存池实例
MemoryPool g_memory_pool;
