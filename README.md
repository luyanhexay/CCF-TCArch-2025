# CCF 2025 HIP Programming Contest - 参赛作品

## 项目概述

本项目是参加 CCF TCARCH- 计算机体系结构挑战赛 2025 的参赛作品，实现了三个基于 AMD ROCm 和 HIP 编程模型的高性能 GPU 算法：

1. **Prefix Sum（前缀和）** - 高效的并行前缀和计算
2. **Softmax** - 数值稳定的 Softmax 函数实现
3. **APSP（全对最短路径）** - 基于分块 Floyd-Warshall 算法的全对最短路径求解

## 技术特点

### 核心优化技术

- **HIP 冷启动优化**：通过多线程并行执行，将 HIP 设备初始化与文件 I/O 操作并行进行，完全隐藏初始化开销
- **内存管理优化**：使用页锁定内存（pinned memory）和异步传输，提高数据传输效率
- **I/O 性能优化**：实现快速读入和带缓冲区的输出函数，减少系统调用开销
- **GPU 并行算法**：针对每个问题设计专门的并行算法，充分利用 GPU 计算资源

### 算法实现

#### 1. Prefix Sum（前缀和）

- **算法**：条带式 warp 优化的块内扫描 + 两段式块间偏移处理
- **优化**：使用共享内存、向量化访存、在线统计算法
- **性能**：在百万级数据规模下，GPU 计算部分耗时 < 10ms

#### 2. Softmax

- **算法**：数值稳定的 Softmax 算法 + 在线统计算法
- **优化**：一次遍历同时计算最大值和求和，避免多次遍历
- **特点**：确保数值稳定性，避免溢出和下溢问题

#### 3. APSP（全对最短路径）

- **算法**：分块 Floyd-Warshall 算法
- **优化**：三阶段计算（对角线块、行列块、剩余块）+ 共享内存优化
- **特点**：提高缓存命中率，减少内存访问延迟

## 项目结构

```
hip_programming_contest/
├── prefix_sum/          # 前缀和实现
│   ├── kernel.hip       # GPU 核函数
│   ├── main.cpp         # 主程序
│   ├── main.h           # 头文件
│   ├── Makefile         # 编译配置
│   ├── README.md        # 详细说明
│   ├── report.md        # 设计报告
│   └── testcases/       # 测试用例
├── softmax/             # Softmax 实现
│   ├── kernel.hip       # GPU 核函数
│   ├── main.cpp         # 主程序
│   ├── main.h           # 头文件
│   ├── Makefile         # 编译配置
│   ├── README.md        # 详细说明
│   ├── report.md        # 设计报告
│   └── testcases/       # 测试用例
├── apsp/                # 全对最短路径实现
│   ├── kernel.hip       # GPU 核函数
│   ├── blocked_floyd_warshall_hip.hip  # 核心算法
│   ├── blocked_floyd_warshall.h        # 算法接口
│   ├── errors.h         # 错误处理
│   ├── main.cpp         # 主程序
│   ├── main.h           # 头文件
│   ├── Makefile         # 编译配置
│   ├── README.md        # 详细说明
│   ├── report.md        # 设计报告
│   └── testcases/       # 测试用例
├── verify.py            # 验证脚本
├── self_test_and_submit.sbatch  # 测试和提交脚本
└── README.md            # 项目总览（本文件）
```

## 编译和运行

### 环境要求

- AMD ROCm 开源堆栈
- HIP 编程模型
- AMD Instinct MI100 GPU（或兼容设备）

### 编译

```bash
# 编译所有项目
cd prefix_sum && make
cd ../softmax && make
cd ../apsp && make
```

### 运行

```bash
# 前缀和
./prefix_sum testcases/1.in

# Softmax
./softmax testcases/1.in

# 全对最短路径
./apsp testcases/1.in
```

### 测试

```bash
# 运行所有测试
python verify.py
```

## 性能表现

### Prefix Sum

- 数据规模：支持最大 10^9 个整数
- 性能：百万级数据 GPU 计算 < 10ms
- 优化效果：相比未优化版本提升 30+ 倍

### Softmax

- 数据规模：支持最大 10^8 个浮点数
- 精度：满足绝对容差 1×10⁻⁶，相对容差 1×10⁻⁵
- 数值稳定性：完全避免溢出和下溢问题

### APSP

- 图规模：支持最大 40,000 个顶点
- 性能：中等规模图（1000 顶点）GPU 计算 < 10ms
- 算法复杂度：O(V³) 时间，O(V²) 空间

## 技术亮点

1. **HIP 冷启动优化**：通过多线程并行执行，将 HIP 初始化与文件 I/O 并行，完全隐藏初始化开销
2. **内存管理优化**：使用页锁定内存和异步传输，显著提高数据传输效率
3. **I/O 性能优化**：实现快速读入和带缓冲区的输出，减少系统调用开销
4. **算法优化**：针对每个问题设计专门的并行算法，充分利用 GPU 计算资源
5. **数值稳定性**：在 Softmax 中实现数值稳定的算法，避免数值计算问题

## 开发团队

本作品由参赛队伍 059 开发，实现了三个高性能的 GPU 算法，在保证正确性的前提下，通过多种优化技术显著提升了程序性能。

## 许可证

本项目仅用于 CCF 2025 HIP Programming Contest 参赛，请勿用于商业用途。

---

**注意**：本项目需要在支持 AMD ROCm 和 HIP 的环境下编译和运行。请确保已正确安装相关开发环境。
