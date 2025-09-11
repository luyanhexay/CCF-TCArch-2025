#!/bin/bash
# 批量参数测试脚本 - 基于softmax.param.md的建议

set -e

echo "================================================="
echo "        Softmax 批量参数优化测试                "
echo "================================================="
echo "开始时间: $(date)"
echo ""

# 测试配置
TEST_CASE="testcases/10.in"  # 使用最大数据集进行测试
RESULTS_DIR="param_test_results"
mkdir -p $RESULTS_DIR

# 1. TPB (Threads Per Block) 测试
echo "1. 测试 TPB (Threads Per Block) 参数..."
echo "-------------------------------------------------"

TPB_VALUES=(64 128 256 512 1024)
for tpb in "${TPB_VALUES[@]}"; do
    echo "测试 TPB=$tpb..."
    job_id=$(sbatch --export=TEST_TYPE=tpb,TPB=$tpb param_test.sbatch | awk '{print $4}')
    echo "  作业ID: $job_id"
    sleep 2  # 避免作业冲突
done

echo "TPB测试作业已提交，等待完成..."
echo ""

# 2. Grid Size Multiplier 测试
echo "2. 测试 Grid Size Multiplier 参数..."
echo "-------------------------------------------------"

GRID_MULTIPLIER_VALUES=(1 2 4 8 16)
for multiplier in "${GRID_MULTIPLIER_VALUES[@]}"; do
    echo "测试 GRID_MULTIPLIER=$multiplier..."
    job_id=$(sbatch --export=TEST_TYPE=grid,GRID_MULTIPLIER=$multiplier param_test.sbatch | awk '{print $4}')
    echo "  作业ID: $job_id"
    sleep 2
done

echo "Grid Multiplier测试作业已提交，等待完成..."
echo ""

# 3. Pinned Memory Threshold 测试
echo "3. 测试 Pinned Memory Threshold 参数..."
echo "-------------------------------------------------"

THRESHOLD_VALUES=(100000 500000 1000000 2000000 5000000)
for threshold in "${THRESHOLD_VALUES[@]}"; do
    echo "测试 PINNED_MEMORY_THRESHOLD=$threshold..."
    job_id=$(sbatch --export=TEST_TYPE=threshold,PINNED_MEMORY_THRESHOLD=$threshold param_test.sbatch | awk '{print $4}')
    echo "  作业ID: $job_id"
    sleep 2
done

echo "Pinned Memory Threshold测试作业已提交，等待完成..."
echo ""

# 4. 组合优化测试
echo "4. 测试最佳参数组合..."
echo "-------------------------------------------------"

# 基于前面的结果，测试最佳组合
echo "测试组合: TPB=128, GRID_MULTIPLIER=4, THRESHOLD=1000000"
job_id=$(sbatch --export=TEST_TYPE=combo,TPB=128,GRID_MULTIPLIER=4,PINNED_MEMORY_THRESHOLD=1000000 param_test.sbatch | awk '{print $4}')
echo "  作业ID: $job_id"

echo ""
echo "所有测试作业已提交!"
echo "使用 'squeue -u $USER' 查看作业状态"
echo "使用 'ls param_test_*.log' 查看结果日志"
echo ""
echo "完成时间: $(date)"
echo "================================================="
