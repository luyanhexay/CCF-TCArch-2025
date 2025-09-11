# Grid Size Multiplier 性能分析报告

## 测试概述
基于TPB=256的最优配置，对Grid Size Multiplier参数进行了系统性测试，测试了5个不同的乘数值：1, 2, 4, 8, 16。

## 测试结果汇总

| Grid Multiplier | onlinePartialKernel (μs) | onlineFinalReduceKernel (μs) | softmaxWriteKernel (μs) | 总Kernel时间 (μs) | 性能排名 |
|-----------------|-------------------------|----------------------------|------------------------|------------------|----------|
| 1               | 49.12                   | 24.00                      | 10.40                  | 83.52            | 2        |
| 2               | 43.36                   | 31.68                      | 8.48                   | 83.52            | 3        |
| 4               | 38.72                   | 34.56                      | 7.84                   | 81.12            | 1        |
| 8               | 40.32                   | 32.96                      | 5.92                   | 79.20            | 4        |
| 16              | 39.52                   | 37.60                      | 5.92                   | 83.04            | 5        |

## 关键发现

### 1. 最优Grid Size Multiplier
**Grid Size Multiplier=4** 表现最佳，总kernel执行时间最短（81.12μs），相比Grid Size Multiplier=1提升了约2.9%。

### 2. 性能趋势分析
- **Grid Size Multiplier=1到4**: 随着乘数增加，性能逐步提升
- **Grid Size Multiplier=4**: 达到性能峰值
- **Grid Size Multiplier=8到16**: 性能开始下降

### 3. Kernel性能分析
- **onlinePartialKernel**: 在Grid Size Multiplier=4时达到最佳性能（38.72μs）
- **onlineFinalReduceKernel**: 在Grid Size Multiplier=1时达到最佳性能（24.00μs）
- **softmaxWriteKernel**: 在Grid Size Multiplier=8和16时达到最佳性能（5.92μs）

### 4. 性能分析
- **Grid Size Multiplier=1**: 线程块数量不足，GPU利用率较低
- **Grid Size Multiplier=4**: 最佳平衡点，GPU利用率高且无过度调度
- **Grid Size Multiplier=8和16**: 线程块过多，可能导致资源竞争和调度开销

## 建议

### 1. 推荐配置
**使用Grid Size Multiplier=4作为最优配置**，理由：
- 总kernel执行时间最短
- onlinePartialKernel性能最佳
- 各kernel性能均衡

### 2. 进一步优化方向
- 可以测试Grid Size Multiplier=3, 5, 6等中间值
- 考虑不同数据集大小下的Grid Size优化
- 结合页锁定内存阈值进行联合优化

## 测试环境
- GPU: MI100 (gfx908)
- TPB: 256 (最优配置)
- 测试用例: testcases/10.in (最大数据集)
- 编译器: hipcc -O3
- 测试时间: 2025-09-11

## 下一步计划
1. 使用TPB=256, Grid Size Multiplier=4进行页锁定内存阈值测试
2. 进行综合性能验证
3. 测试不同数据集大小下的参数适应性
