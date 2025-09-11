# APSP重构后性能分析报告

## 测试环境
- **小数据集**: testcases/10.in (128节点)
- **大数据集**: testcases/7.in (6400节点) 
- **GPU**: MI100 (gfx908)
- **工具**: rocprof
- **架构**: APSP_Solver重构版本

## 性能对比总结

### 执行时间对比
| 数据集 | 重构前 | 重构后 | 变化 |
|--------|--------|--------|------|
| 小数据集(128节点) | 0.65s | 0.52s | ⬇️ **20%提升** |
| 大数据集(6400节点) | 1.91s | 2.12s | ⬆️ **11%增加** |

### 小数据集详细分析 (testcases/10.in)

**HIP API时间分布:**
```
hipMemcpy:               484.1ms (99.70%) - 主要瓶颈
hipLaunchKernel:         0.68ms  (0.14%)  - Kernel启动
hipHostMalloc:           0.42ms  (0.09%)  - 页锁定内存分配
hipHostFree:             0.20ms  (0.04%)  - 页锁定内存释放
其他API:                 <0.01%           - 配置和错误检查
```

**Kernel执行时间分布:**
```
floyd_warshall_block_kernel_phase1:     82.0ms (55.69%) - 对角线块更新
floyd_warshall_block_kernel_phase2_col:  65.0ms (44.13%) - 列更新  
floyd_warshall_block_kernel_phase3:      0.16ms (0.11%) - 剩余块更新
floyd_warshall_block_kernel_phase2_row:  0.11ms (0.07%) - 行更新
总Kernel时间:                           147.2ms
```

## 关键发现

### ✅ 重构成功方面
1. **小数据集性能提升20%**: 从650ms降至520ms
2. **页锁定内存已启用**: 使用hipHostMalloc替代普通内存
3. **架构更清晰**: APSP_Solver类管理GPU资源
4. **错误处理完善**: 更好的内存管理和错误报告

### ⚠️ 当前问题
1. **solve()函数仍有数据传输**: 
   - 当前实现：GPU → 页锁定内存 → blocked_floyd_warshall → 页锁定内存 → GPU
   - 这导致了额外的数据传输开销

2. **Kernel时间异常增加**:
   - Phase1: 从39.5μs增加到82.0ms (2075x增加)
   - Phase2_col: 从38.7μs增加到65.0ms (1679x增加)
   - 原因：使用了页锁定内存作为中介，而非纯GPU计算

3. **大数据集性能下降**:
   - 从1.91s增加到2.12s
   - 额外数据传输开销在大数据集上更明显

## 下一步优化重点

### 🎯 第二步：实现纯GPU计算
**目标**: 消除solve()中的GPU↔Host数据传输

**当前问题代码**:
```cpp
// solve()函数中的问题
HIP_CHECK(hipMemcpy(h_graph_pinned, d_graph, size, hipMemcpyDeviceToHost));
blocked_floyd_warshall(h_graph_pinned, V, block_size, h_result_pinned, &err);
HIP_CHECK(hipMemcpy(d_graph, h_result_pinned, size, hipMemcpyHostToDevice));
```

**解决方案**:
1. 创建GPU版本的blocked_floyd_warshall函数
2. 直接在GPU内存上执行kernel
3. 消除所有GPU↔Host数据传输

**预期效果**:
- 小数据集: 520ms → 100-200ms (减少60-80%)
- 大数据集: 2.12s → 800-1200ms (减少40-60%)

### 🔧 第三步：Kernel优化
1. **向量化内存访问**: 使用int4类型减少内存访问次数
2. **增加线程并行度**: 每个线程处理多个元素
3. **优化phase3 kernel**: 大数据集的主要计算瓶颈

## 总结

**第一步架构重构已完成** ✅
- 成功实现APSP_Solver类
- 启用页锁定内存优化
- 小数据集性能提升20%

**下一步重点** 🎯
- 实现纯GPU计算，消除solve()中的数据传输
- 预期大幅提升性能，特别是大数据集

重构的基础已经打好，现在需要继续实施第二步优化来实现真正的纯GPU计算。

---
**报告时间**: 2025年1月11日  
**优化状态**: 第一步完成，准备第二步优化

## 大数据集详细分析 (testcases/7.in)

### HIP API时间分布

**重构后 (6400节点):**
```
hipMemcpy:               720.8ms (91.36%) - 主要瓶颈
hipHostMalloc:           60.7ms  (7.70%)  - 页锁定内存分配开销
hipLaunchKernel:         4.2ms   (0.53%)  - Kernel启动开销
hipHostFree:             1.5ms   (0.19%)  - 页锁定内存释放
hipFree:                 1.0ms   (0.13%)  - GPU内存释放
hipMalloc:               0.18ms  (0.02%)  - GPU内存分配
其他API:                 <0.01%           - 配置和错误检查
```

### Kernel执行时间分布

**重构后 (6400节点):**
```
floyd_warshall_block_kernel_phase3:     336.8ms (97.84%) - 剩余块更新
floyd_warshall_block_kernel_phase2_row:  2.83ms  (0.82%) - 行更新
floyd_warshall_block_kernel_phase2_col:  2.64ms  (0.77%) - 列更新
floyd_warshall_block_kernel_phase1:      1.96ms  (0.57%) - 对角线块更新
总Kernel时间:                           343.2ms
```

### 大数据集性能对比

| 指标 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| 总执行时间 | 1.91s | 2.12s | ⬆️ 11% |
| hipMemcpy时间 | 734ms (38.4%) | 721ms (91.4%) | ⬇️ 2% |
| Kernel总时间 | 344ms (18.0%) | 343ms (16.2%) | ⬇️ 0.3% |
| hipHostMalloc开销 | 0ms | 61ms (7.7%) | ⬆️ 新增 |

### 大数据集关键发现

1. **页锁定内存分配开销显著**: 
   - hipHostMalloc占用7.7%的时间 (61ms)
   - 这是重构引入的新开销

2. **Kernel性能基本保持**:
   - Phase3 kernel仍然是主要瓶颈 (97.84%)
   - 总Kernel时间基本不变 (343ms vs 344ms)

3. **内存传输效率提升**:
   - hipMemcpy时间略有减少 (734ms → 721ms)
   - 但占总时间的比例增加 (38.4% → 91.4%)

## 综合性能分析

### 重构效果评估

**积极影响 ✅:**
- 小数据集性能提升20%
- 页锁定内存优化开始发挥作用
- 代码架构更清晰

**需要改进 ⚠️:**
- 大数据集性能略有下降
- hipHostMalloc开销显著
- solve()函数中仍有不必要的数据传输

### 下一步优化重点

1. **消除solve()中的数据传输** (最高优先级)
2. **优化hipHostMalloc开销**
3. **实现纯GPU计算**
