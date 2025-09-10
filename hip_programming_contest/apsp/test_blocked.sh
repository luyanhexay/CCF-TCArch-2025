#!/bin/bash
# 简单的本地测试脚本，用于测试分块Floyd-Warshall实现

echo "=========================================="
echo "APSP Blocked Floyd-Warshall Local Test"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "kernel_blocked.hip" ]; then
    echo "ERROR: Please run this script from the apsp directory"
    exit 1
fi

# 编译代码
echo "Compiling blocked Floyd-Warshall implementation..."
make -f Makefile_blocked clean
make -f Makefile_blocked apsp_blocked

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# 检查测试用例
if [ ! -d "testcases" ] || [ -z "$(ls testcases/*.in 2>/dev/null)" ]; then
    echo "WARNING: No test cases found in testcases/ directory"
    echo "Creating a simple test case..."
    
    # 创建测试用例目录
    mkdir -p testcases
    
    # 创建一个简单的测试用例
    cat > testcases/simple.in << EOF
3 3
0 1 1
1 2 2
0 2 4
EOF
    
    # 创建对应的期望输出
    cat > testcases/simple.out << EOF
0 1 3
1073741823 0 2
1073741823 1073741823 0
EOF
fi

# 运行测试
echo "Running test cases..."
echo "=========================================="

for input_file in testcases/*.in; do
    if [ -f "$input_file" ]; then
        test_name=$(basename "$input_file" .in)
        echo "Testing: $test_name"
        
        # 运行分块版本
        echo "Running blocked Floyd-Warshall..."
        start_time=$(date +%s.%N)
        ./apsp_blocked "$input_file" > "output_${test_name}.out"
        exit_code=$?
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        
        if [ $exit_code -eq 0 ]; then
            echo "  ✓ Execution successful"
            echo "  ✓ Execution time: ${duration} seconds"
            
            # 检查输出格式
            output_lines=$(wc -l < "output_${test_name}.out")
            echo "  ✓ Output lines: $output_lines"
            
            # 如果有期望输出，进行比较
            expected_file="testcases/${test_name}.out"
            if [ -f "$expected_file" ]; then
                if diff "output_${test_name}.out" "$expected_file" > /dev/null; then
                    echo "  ✓ Output matches expected result"
                else
                    echo "  ✗ Output differs from expected result"
                    echo "  Expected:"
                    cat "$expected_file"
                    echo "  Got:"
                    cat "output_${test_name}.out"
                fi
            else
                echo "  ℹ No expected output file found, showing result:"
                cat "output_${test_name}.out"
            fi
        else
            echo "  ✗ Execution failed with exit code: $exit_code"
        fi
        
        echo "----------------------------------------"
    fi
done

# 清理输出文件
echo "Cleaning up output files..."
rm -f output_*.out

echo "=========================================="
echo "Test completed!"
echo "=========================================="

# 显示使用说明
echo ""
echo "Usage instructions:"
echo "  - To run a specific test: ./apsp_blocked testcases/<test_name>.in"
echo "  - To compare with original: make -f Makefile_blocked compare"
echo "  - To run performance benchmark: make -f Makefile_blocked benchmark"
echo "  - To submit to cluster: sbatch self_test_apsp_blocked.sbatch"
