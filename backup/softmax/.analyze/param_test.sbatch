#!/bin/bash
#SBATCH -J Softmax_ParamTest
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o param_test_%j.log
#SBATCH --mem=32G
#SBATCH -D /home/user059/hip_programming_contest/softmax

# 参数化测试脚本 - 基于softmax.param.md的建议
# 用法: sbatch --export=TEST_TYPE=tpb,TPB=256 param_test.sbatch

set -e

# 默认参数
TEST_TYPE=${TEST_TYPE:-"tpb"}
TPB=${TPB:-128}
GRID_MULTIPLIER=${GRID_MULTIPLIER:-4}
PINNED_MEMORY_THRESHOLD=${PINNED_MEMORY_THRESHOLD:-1000000}
TEST_CASE=${TEST_CASE:-"testcases/10.in"}

echo "================================================="
echo "           Softmax 参数化测试                    "
echo "================================================="
echo "测试类型: $TEST_TYPE"
echo "测试用例: $TEST_CASE"
echo "TPB: $TPB"
echo "GRID_MULTIPLIER: $GRID_MULTIPLIER"
echo "PINNED_MEMORY_THRESHOLD: $PINNED_MEMORY_THRESHOLD"
echo "开始时间: $(date)"
echo ""

# 清理旧文件
rm -f softmax
rm -rf my_outputs
mkdir -p my_outputs

# 编译
echo "编译参数化版本..."
make clean
make TPB=$TPB GRID_MULTIPLIER=$GRID_MULTIPLIER PINNED_MEMORY_THRESHOLD=$PINNED_MEMORY_THRESHOLD

if [ ! -f ./softmax ]; then
    echo "编译失败!"
    exit 1
fi

echo "编译成功"
echo ""

# 运行测试
echo "运行性能测试..."
echo "测试用例: $TEST_CASE"
echo "-------------------------------------------------"

# 使用rocprof进行性能分析
rocprof --hip-trace --timestamp on --stats ./softmax "$TEST_CASE" > my_outputs/output.txt 2>&1

# 提取关键性能指标
if [ -f results.stats.csv ]; then
    echo "Kernel执行时间分析:"
    echo "-------------------------------------------------"
    # 提取kernel时间
    grep -E "(onlinePartialKernel|onlineFinalReduceKernel|softmaxWriteKernel)" results.stats.csv | \
    awk -F',' '{printf "%-25s: %8.2fμs\n", $1, $2*1000000}' || echo "未找到kernel统计信息"
    echo ""
fi

if [ -f results.hip_stats.csv ]; then
    echo "HIP API时间分析:"
    echo "-------------------------------------------------"
    # 提取API时间
    awk -F',' 'NR>1 {printf "%-20s: %8.2fms (%5.1f%%)\n", $1, $2*1000, $3}' results.hip_stats.csv | \
    head -10 || echo "未找到API统计信息"
    echo ""
fi

# 计算总执行时间
TOTAL_TIME=$(grep "Total execution time" my_outputs/output.txt | awk '{print $4}' | sed 's/s//' || echo "N/A")
echo "总执行时间: $TOTAL_TIME"
echo ""

# 验证正确性
echo "验证正确性..."
BASE_NAME=$(basename "$TEST_CASE" .in)
GOLDEN_FILE="testcases/${BASE_NAME}.out"
MY_OUTPUT="my_outputs/${BASE_NAME}.myout"

if [ -f "$GOLDEN_FILE" ]; then
    if python3 /home/user059/hip_programming_contest/verify.py "$MY_OUTPUT" "$GOLDEN_FILE" >/dev/null 2>&1; then
        echo "正确性验证: PASS"
    else
        echo "正确性验证: FAIL"
        echo "输出文件: $MY_OUTPUT"
        echo "标准答案: $GOLDEN_FILE"
    fi
else
    echo "警告: 未找到标准答案文件 $GOLDEN_FILE"
fi

echo ""
echo "测试完成时间: $(date)"
echo "================================================="
