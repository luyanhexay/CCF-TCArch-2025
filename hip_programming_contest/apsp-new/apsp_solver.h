#ifndef APSP_SOLVER_H
#define APSP_SOLVER_H

#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>

/**
 * APSP_Solver类 - 管理GPU资源的持久化计算环境
 * 
 * 设计目标：
 * 1. 一次性初始化：将图数据从主机拷贝到设备
 * 2. 纯设备计算：在GPU上执行Floyd-Warshall算法，无主机-设备数据交换
 * 3. 结果获取：计算完成后将结果从设备拷贝回主机
 */
class APSP_Solver {
private:
    // GPU设备内存
    int* d_graph;              // GPU上的图数据
    size_t allocated_size;     // 已分配的GPU内存大小
    int current_V;             // 当前图的节点数
    int current_block_size;    // 当前使用的块大小
    
    // 页锁定主机内存
    int* h_graph_pinned;       // 页锁定的输入图内存
    int* h_result_pinned;      // 页锁定的结果内存
    size_t pinned_size;        // 页锁定内存大小
    
    
    // 状态标志
    bool initialized;          // 是否已初始化
    bool graph_loaded;         // 图数据是否已加载到GPU
    
    // 错误处理
    struct SolverError {
        int code;
        char message[256];
    } last_error;
    
    // 内部方法
    void set_error(int code, const char* message);
    int allocate_gpu_memory(int V);
    int allocate_pinned_memory(int V);
    void cleanup_memory();
    
public:
    APSP_Solver();
    ~APSP_Solver();
    
    // 禁用拷贝构造和赋值
    APSP_Solver(const APSP_Solver&) = delete;
    APSP_Solver& operator=(const APSP_Solver&) = delete;
    
    /**
     * 初始化求解器
     * @param graph 输入图的邻接矩阵 (V x V)
     * @param V 图的节点数
     * @param block_size 块大小 (默认16)
     * @return 0成功，非0失败
     */
    int init(const int* graph, int V, int block_size = 16);
    
    /**
     * 在GPU上执行APSP计算
     * @return 0成功，非0失败
     */
    int solve();
    
    /**
     * 获取计算结果
     * @param result 输出缓冲区 (V x V)
     * @return 0成功，非0失败
     */
    int get_result(int* result);
    
    /**
     * 检查是否已初始化
     */
    bool is_initialized() const { return initialized; }
    
    /**
     * 获取当前图的节点数
     */
    int get_vertex_count() const { return current_V; }
    
    /**
     * 获取最后的错误信息
     */
    const char* get_last_error() const { return last_error.message; }
    
    /**
     * 重置求解器状态（释放内存，准备重新初始化）
     */
    void reset();
};

// GPU kernel函数声明
extern "C" int apsp_solver_solve_gpu(int V, int oversized_n, int block_size, int* d_graph);

#endif // APSP_SOLVER_H
