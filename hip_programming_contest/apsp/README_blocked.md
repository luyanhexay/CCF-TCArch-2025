# 分块 Floyd-Warshall 算法 HIP 实现

这是基于 CUDA 分块 Floyd-Warshall 算法移植到 HIP 环境的实现，用于与原始实现进行性能对比。

## 文件结构

### 新增文件

- `kernel_blocked.hip` - 分块 Floyd-Warshall 算法的 HIP kernel 实现
- `main_blocked.cpp` - 分块版本的主程序
- `blocked_floyd_warshall.h` - 分块算法的头文件
- `Makefile_blocked` - 分块版本的编译配置
- `self_test_apsp_blocked.sbatch` - 分块版本的集群测试脚本
- `compare_implementations.sbatch` - 两个版本的对比测试脚本
- `test_blocked.sh` - 本地测试脚本
- `README_blocked.md` - 本说明文件

### 原始文件

- `kernel.hip` - 原始 Floyd-Warshall 算法实现
- `main.cpp` - 原始版本的主程序
- `main.h` - 原始版本的头文件
- `Makefile` - 原始版本的编译配置
- `self_test_apsp.sbatch` - 原始版本的集群测试脚本

## 算法特点

### 分块 Floyd-Warshall 算法

- **三阶段优化**：
  1. Phase 1: 更新对角线块 (k,k)
  2. Phase 2: 更新行块 (k,j) 和列块 (i,k)
  3. Phase 3: 更新剩余块 (i,j)
- **内存优化**：使用共享内存减少全局内存访问
- **并行优化**：充分利用 GPU 的并行计算能力

### 动态块大小选择

根据图的大小自动选择最优块大小：

- V ≤ 1000: 块大小 16
- V ≤ 5000: 块大小 32
- V ≤ 20000: 块大小 64
- V > 20000: 块大小 128

## 编译和使用

### 编译分块版本

```bash
make -f Makefile_blocked apsp_blocked
```

### 编译原始版本

```bash
make apsp
```

### 编译两个版本

```bash
make -f Makefile_blocked all
```

### 运行测试

```bash
# 本地测试分块版本
./test_blocked.sh

# 测试特定文件
./apsp_blocked testcases/1.in

# 对比两个版本
make -f Makefile_blocked compare

# 性能基准测试
make -f Makefile_blocked benchmark
```

### 集群测试

```bash
# 测试分块版本
sbatch self_test_apsp_blocked.sbatch

# 对比两个版本
sbatch compare_implementations.sbatch
```

## 性能对比

### 预期性能特点

- **稠密图**：分块版本通常性能更优
- **稀疏图**：原始版本可能更优
- **大规模图**：分块版本的内存访问模式更优

### 测试指标

- 执行时间
- 内存使用
- 结果正确性
- 加速比

## 调试和优化

### 调试选项

在编译时添加调试标志：

```bash
make -f Makefile_blocked HIP_FLAGS="-g -O0" apsp_blocked
```

### 性能分析

使用 HIP 性能分析工具：

```bash
rocprof --hip-trace ./apsp_blocked testcases/1.in
```

### 常见问题

1. **编译错误**：检查 HIP 环境是否正确加载
2. **运行时错误**：检查 GPU 内存是否足够
3. **结果不匹配**：检查块大小设置是否合适

## 扩展功能

### 自定义块大小

在`main_blocked.cpp`中修改块大小选择逻辑：

```cpp
int block_size = 32; // 自定义块大小
```

### 添加更多优化

- 使用纹理内存
- 实现多 GPU 版本
- 添加内存预取优化

## 参考资源

- [HIP 编程指南](https://rocm.docs.amd.com/projects/HIP/)
- [Floyd-Warshall 算法](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)
- [GPU 并行算法优化](https://developer.nvidia.com/gpu-accelerated-libraries)

## 注意事项

1. 确保 HIP 环境正确配置
2. 测试前检查 GPU 内存是否足够
3. 大规模图测试时注意内存使用
4. 集群测试时注意作业队列限制
