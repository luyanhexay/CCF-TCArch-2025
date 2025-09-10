# 快速启动指南

## 概述

本项目包含两个并行的 APSP 实现：

1. **原始实现** (`kernel.hip`, `main.cpp`) - 基础 Floyd-Warshall 算法
2. **分块实现** (`kernel_blocked.hip`, `main_blocked.cpp`) - 优化的分块 Floyd-Warshall 算法

## 快速开始

### 1. 编译两个版本

```bash
# 编译原始版本
make apsp

# 编译分块版本
make -f Makefile_blocked apsp_blocked

# 或者一次性编译两个版本
make -f Makefile_blocked all
```

### 2. 运行测试

```bash
# 本地快速测试
./test_blocked.sh

# 测试特定文件
./apsp testcases/1.in
./apsp_blocked testcases/1.in

# 对比两个版本
make -f Makefile_blocked compare
```

### 3. 集群测试

```bash
# 测试分块版本
sbatch self_test_apsp_blocked.sbatch

# 对比两个版本
sbatch compare_implementations.sbatch
```

## 文件说明

### 核心文件

- `kernel.hip` / `kernel_blocked.hip` - GPU kernel 实现
- `main.cpp` / `main_blocked.cpp` - 主程序
- `main.h` / `blocked_floyd_warshall.h` - 头文件

### 构建文件

- `Makefile` - 原始版本构建配置
- `Makefile_blocked` - 分块版本构建配置

### 测试文件

- `test_blocked.sh` - 本地测试脚本
- `self_test_apsp.sbatch` - 原始版本集群测试
- `self_test_apsp_blocked.sbatch` - 分块版本集群测试
- `compare_implementations.sbatch` - 对比测试

## 性能对比

### 预期结果

- **小图 (V < 1000)**: 两种实现性能相近
- **中图 (1000 ≤ V < 10000)**: 分块版本开始显示优势
- **大图 (V ≥ 10000)**: 分块版本明显更优

### 关键指标

- 执行时间
- 内存使用效率
- GPU 利用率
- 结果正确性

## 故障排除

### 编译问题

```bash
# 检查HIP环境
which hipcc
hipcc --version

# 清理重新编译
make clean
make -f Makefile_blocked clean
```

### 运行时问题

```bash
# 检查GPU状态
rocm-smi

# 检查内存使用
free -h
```

### 结果验证

```bash
# 对比输出
diff output1.txt output2.txt

# 检查输出格式
head -5 output.txt
```

## 下一步

1. **性能调优**: 根据测试结果调整块大小
2. **算法优化**: 实现更多优化策略
3. **扩展功能**: 添加多 GPU 支持
4. **文档完善**: 添加更多使用示例

## 联系信息

如有问题，请参考：

- `README_blocked.md` - 详细文档
- 代码注释
- 测试用例
