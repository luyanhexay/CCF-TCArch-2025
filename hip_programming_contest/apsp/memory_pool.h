#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <hip/hip_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>

/**
 * 内存池类 - 优化GPU和页锁定内存分配
 * 
 * 设计目标：
 * 1. 预分配常用大小的内存块
 * 2. 重用已分配的内存，避免频繁分配/释放
 * 3. 支持GPU内存和页锁定内存的池化管理
 */
class MemoryPool {
private:
    // GPU内存池
    struct GPUBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    // 页锁定内存池
    struct PinnedBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<GPUBlock> gpu_blocks;
    std::vector<PinnedBlock> pinned_blocks;
    std::mutex pool_mutex;
    
    // 预定义的内存大小（基于常见测试用例）
    static constexpr size_t COMMON_SIZES[] = {
        128 * 128 * sizeof(int),      // 128节点 - 64KB
        256 * 256 * sizeof(int),      // 256节点 - 256KB
        512 * 512 * sizeof(int),      // 512节点 - 1MB
        1024 * 1024 * sizeof(int),    // 1024节点 - 4MB
        2048 * 2048 * sizeof(int),    // 2048节点 - 16MB
        4096 * 4096 * sizeof(int),    // 4096节点 - 64MB
        6400 * 6400 * sizeof(int),    // 6400节点 - 163MB
        8192 * 8192 * sizeof(int),   // 8192节点 - 256MB
        10240 * 10240 * sizeof(int),  // 10240节点 - 400MB
        12288 * 12288 * sizeof(int)   // 12288节点 - 576MB
    };
    
    static constexpr size_t NUM_COMMON_SIZES = sizeof(COMMON_SIZES) / sizeof(COMMON_SIZES[0]);
    
    // 内部方法
    void* find_or_allocate_gpu_block(size_t size);
    void* find_or_allocate_pinned_block(size_t size);
    void preallocate_common_sizes();
    void preallocate_small_sizes(); // 只预分配小内存块
    void* allocate_large_block(size_t size); // 大内存块延迟分配
    
public:
    MemoryPool();
    ~MemoryPool();
    
    // 禁用拷贝构造和赋值
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    /**
     * 获取GPU内存块
     * @param size 所需内存大小
     * @return GPU内存指针，失败返回nullptr
     */
    void* get_gpu_memory(size_t size);
    
    /**
     * 获取页锁定内存块
     * @param size 所需内存大小
     * @return 页锁定内存指针，失败返回nullptr
     */
    void* get_pinned_memory(size_t size);
    
    /**
     * 释放GPU内存块（标记为可用，不实际释放）
     * @param ptr GPU内存指针
     */
    void release_gpu_memory(void* ptr);
    
    /**
     * 释放页锁定内存块（标记为可用，不实际释放）
     * @param ptr 页锁定内存指针
     */
    void release_pinned_memory(void* ptr);
    
    /**
     * 获取内存池统计信息
     */
    void get_stats(size_t& gpu_blocks_count, size_t& pinned_blocks_count, 
                   size_t& gpu_total_size, size_t& pinned_total_size);
    
    /**
     * 清理所有内存（仅在程序结束时调用）
     */
    void cleanup_all();
};

// 全局内存池实例
extern MemoryPool g_memory_pool;

#endif // MEMORY_POOL_H
