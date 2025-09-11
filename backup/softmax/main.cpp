#include "main.h"
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <thread>

// --- HIP Warmup Functions ---
__global__ void warmup_kernel() {}

static void hip_warmup_once()
{
    // 1) 提前创建上下文
    hipFree(0);                    // 常用的"唤醒设备"手段
    // 2) 提前创建 stream（如需）
    hipStream_t s; hipStreamCreate(&s);
    // 3) 发一个极小的 kernel，确保 code object 装载到驱动
    hipLaunchKernelGGL(warmup_kernel, dim3(1), dim3(1), 0, s);
    hipStreamSynchronize(s);
    hipStreamDestroy(s);
    // 此刻：上下文/模块已就绪，后续真正的 kernel 不再付这笔代价
}

// --- Fast I/O Functions (Buffered Output) ---
const int OUT_BUFFER_SIZE = 1 << 20; // 1MB buffer
char out_buffer[OUT_BUFFER_SIZE];
int out_pos = 0;

inline void write_char(char ch) {
    if (out_pos == OUT_BUFFER_SIZE) {
        fwrite(out_buffer, 1, OUT_BUFFER_SIZE, stdout);
        out_pos = 0;
    }
    out_buffer[out_pos++] = ch;
}

void write_flush() {
    if (out_pos > 0) {
        fwrite(out_buffer, 1, out_pos, stdout);
        out_pos = 0;
    }
}

inline void write_float(float x) {
    // Use snprintf to format the float into a temporary buffer
    // and then write it to the main output buffer.
    // "%g" is a good general-purpose format specifier for floats,
    // balancing precision and size.
    static char temp_buffer[64];
    int len = snprintf(temp_buffer, sizeof(temp_buffer), "%.6g", x);
    for (int i = 0; i < len; ++i) {
        write_char(temp_buffer[i]);
    }
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    // 1. Argument and file check
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // 启动HIP预热线程（与文件I/O并行）
    std::thread hip_init_thread([]{
        hip_warmup_once();
    });

    // 2. Redirect standard input from the specified file for faster reading
    if (freopen(argv[1], "r", stdin) == NULL) {
        std::cerr << "fileopen error: " << argv[1] << std::endl;
        return 1;
    }

    // 3. Optional: Disable C++ stream synchronization
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // 4. Read input using scanf for speed
    int N;
    if (scanf("%d", &N) != 1) {
        std::cerr << "error reading N" << std::endl;
        return 1;
    }

    // 5. Read N floating-point numbers
    std::vector<float> input(N);
    for (int i = 0; i < N; ++i) {
        if (scanf("%f", &input[i]) != 1) {
            std::cerr << "error reading input float at index " << i << std::endl;
            return 1;
        }
    }

    // 等 HIP 初始化线程收尾（如果它比 I/O 慢，就等待；如果它更快，则已隐藏）
    hip_init_thread.join();

    // 6. Allocate output vector and call the GPU solver
    std::vector<float> output(N);
    solve(input.data(), output.data(), N);

    // 7. Write output using buffered fast I/O
    for (int i = 0; i < N; ++i) {
        write_float(output[i]);
        if (i < N - 1) {
            write_char(' ');
        }
    }
    write_char('\n');

    // 8. Flush the output buffer
    write_flush();

    return 0;
}