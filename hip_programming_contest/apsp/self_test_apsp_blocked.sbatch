#!/bin/bash
#SBATCH --job-name=apsp_blocked_test
#SBATCH --output=apsp_blocked_test_%j.out
#SBATCH --error=apsp_blocked_test_%j.err
#SBATCH --time=00:10:00
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
echo "APSP Blocked Floyd-Warshall Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $HIP_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

# 编译代码
echo "Compiling blocked Floyd-Warshall implementation..."
make -f Makefile_blocked clean
make -f Makefile_blocked $(TARGET_BLOCKED)

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# 运行测试用例
echo "Running test cases..."
echo "=========================================="

# 测试所有输入文件
for input_file in testcases/*.in; do
    if [ -f "$input_file" ]; then
        test_name=$(basename "$input_file" .in)
        echo "Testing: $test_name"
        echo "Input file: $input_file"
        
        # 运行分块版本
        echo "Running blocked Floyd-Warshall..."
        start_time=$(date +%s.%N)
        ./apsp_blocked "$input_file" > "output_blocked_${test_name}.out"
        blocked_exit_code=$?
        end_time=$(date +%s.%N)
        blocked_duration=$(echo "$end_time - $start_time" | bc)
        
        if [ $blocked_exit_code -eq 0 ]; then
            echo "  ✓ Blocked version completed successfully"
            echo "  ✓ Execution time: ${blocked_duration} seconds"
        else
            echo "  ✗ Blocked version failed with exit code: $blocked_exit_code"
        fi
        
        # 检查输出文件
        if [ -f "output_blocked_${test_name}.out" ]; then
            output_lines=$(wc -l < "output_blocked_${test_name}.out")
            echo "  ✓ Output file generated with $output_lines lines"
        else
            echo "  ✗ No output file generated"
        fi
        
        echo "----------------------------------------"
    fi
done

# 性能统计
echo "Performance Summary:"
echo "=========================================="

# 计算总测试时间
total_tests=$(ls testcases/*.in 2>/dev/null | wc -l)
echo "Total test cases: $total_tests"

# 检查所有输出文件
successful_tests=0
for input_file in testcases/*.in; do
    if [ -f "$input_file" ]; then
        test_name=$(basename "$input_file" .in)
        if [ -f "output_blocked_${test_name}.out" ]; then
            successful_tests=$((successful_tests + 1))
        fi
    fi
done

echo "Successful tests: $successful_tests/$total_tests"

# 清理临时文件（可选）
# rm -f output_blocked_*.out

echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="

# 显示GPU使用情况
echo "GPU Information:"
rocm-smi --showid

exit 0
