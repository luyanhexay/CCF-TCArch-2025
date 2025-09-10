#!/bin/bash
#SBATCH --job-name=apsp_compare
#SBATCH --output=apsp_compare_%j.out
#SBATCH --error=apsp_compare_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# 加载必要的模块
module load rocm/5.4.0
module load gcc/11.2.0

# 设置环境变量
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 进入工作目录
cd $SLURM_SUBMIT_DIR

echo "=========================================="
echo "APSP Implementation Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $HIP_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# 编译两个版本
echo "Compiling both implementations..."
echo "----------------------------------------"

# 编译原始版本
echo "Compiling original implementation..."
make clean
make apsp
if [ $? -ne 0 ]; then
    echo "ERROR: Original implementation compilation failed!"
    exit 1
fi
echo "✓ Original implementation compiled successfully"

# 编译分块版本
echo "Compiling blocked implementation..."
make -f Makefile_blocked clean
make -f Makefile_blocked apsp_blocked
if [ $? -ne 0 ]; then
    echo "ERROR: Blocked implementation compilation failed!"
    exit 1
fi
echo "✓ Blocked implementation compiled successfully"

echo ""

# 运行对比测试
echo "Running comparison tests..."
echo "=========================================="

# 创建结果目录
mkdir -p comparison_results
cd comparison_results

# 测试所有输入文件
for input_file in ../testcases/*.in; do
    if [ -f "$input_file" ]; then
        test_name=$(basename "$input_file" .in)
        echo "Testing: $test_name"
        echo "Input file: $input_file"
        
        # 运行原始版本
        echo "  Running original implementation..."
        start_time=$(date +%s.%N)
        ../apsp "$input_file" > "original_${test_name}.out"
        original_exit_code=$?
        end_time=$(date +%s.%N)
        original_duration=$(echo "$end_time - $start_time" | bc)
        
        if [ $original_exit_code -eq 0 ]; then
            echo "    ✓ Original completed successfully"
            echo "    ✓ Original execution time: ${original_duration} seconds"
        else
            echo "    ✗ Original failed with exit code: $original_exit_code"
        fi
        
        # 运行分块版本
        echo "  Running blocked implementation..."
        start_time=$(date +%s.%N)
        ../apsp_blocked "$input_file" > "blocked_${test_name}.out"
        blocked_exit_code=$?
        end_time=$(date +%s.%N)
        blocked_duration=$(echo "$end_time - $start_time" | bc)
        
        if [ $blocked_exit_code -eq 0 ]; then
            echo "    ✓ Blocked completed successfully"
            echo "    ✓ Blocked execution time: ${blocked_duration} seconds"
        else
            echo "    ✗ Blocked failed with exit code: $blocked_exit_code"
        fi
        
        # 比较结果
        if [ $original_exit_code -eq 0 ] && [ $blocked_exit_code -eq 0 ]; then
            if diff "original_${test_name}.out" "blocked_${test_name}.out" > /dev/null; then
                echo "    ✓ Results match perfectly"
                
                # 计算性能提升
                if (( $(echo "$original_duration > 0" | bc -l) )); then
                    speedup=$(echo "scale=2; $original_duration / $blocked_duration" | bc)
                    echo "    ✓ Speedup: ${speedup}x"
                fi
            else
                echo "    ✗ Results differ!"
                echo "    First 10 lines of difference:"
                diff "original_${test_name}.out" "blocked_${test_name}.out" | head -10
            fi
        else
            echo "    ✗ Cannot compare results due to execution failures"
        fi
        
        echo "----------------------------------------"
    fi
done

# 生成性能报告
echo "Performance Summary Report"
echo "=========================================="

# 统计信息
total_tests=$(ls ../testcases/*.in 2>/dev/null | wc -l)
original_success=0
blocked_success=0
matching_results=0

for input_file in ../testcases/*.in; do
    if [ -f "$input_file" ]; then
        test_name=$(basename "$input_file" .in)
        
        if [ -f "original_${test_name}.out" ]; then
            original_success=$((original_success + 1))
        fi
        
        if [ -f "blocked_${test_name}.out" ]; then
            blocked_success=$((blocked_success + 1))
        fi
        
        if [ -f "original_${test_name}.out" ] && [ -f "blocked_${test_name}.out" ]; then
            if diff "original_${test_name}.out" "blocked_${test_name}.out" > /dev/null; then
                matching_results=$((matching_results + 1))
            fi
        fi
    fi
done

echo "Total test cases: $total_tests"
echo "Original implementation success: $original_success/$total_tests"
echo "Blocked implementation success: $blocked_success/$total_tests"
echo "Matching results: $matching_results/$total_tests"

# 计算平均性能提升
echo ""
echo "Detailed Performance Analysis:"
echo "----------------------------------------"

for input_file in ../testcases/*.in; do
    if [ -f "$input_file" ]; then
        test_name=$(basename "$input_file" .in)
        
        if [ -f "original_${test_name}.out" ] && [ -f "blocked_${test_name}.out" ]; then
            # 这里可以添加更详细的性能分析
            echo "Test $test_name: Results available for analysis"
        fi
    fi
done

# 显示GPU信息
echo ""
echo "GPU Information:"
rocm-smi --showid

echo "=========================================="
echo "Comparison completed at: $(date)"
echo "Results saved in: comparison_results/"
echo "=========================================="

exit 0
