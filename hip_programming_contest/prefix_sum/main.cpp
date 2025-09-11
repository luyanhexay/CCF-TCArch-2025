#include "main.h"
#include <cstdio>

__global__ void warmup_kernel() {}

static void hip_warmup_once()
{
    // 1) 提前创建上下文
    hipFree(0);                    // 常用的“唤醒设备”手段
    // 2) 提前创建 stream（如需）
    hipStream_t s; hipStreamCreate(&s);
    // 3) 发一个极小的 kernel，确保 code object 装载到驱动
    hipLaunchKernelGGL(warmup_kernel, dim3(1), dim3(1), 0, s);
    hipStreamSynchronize(s);
    hipStreamDestroy(s);
    // 此刻：上下文/模块已就绪，后续真正的 kernel 不再付这笔代价
}


// --- Fast I/O for Integers ---
// Uses getchar() for fast reading of numbers.
inline int read_int() {
    int x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + (ch - '0');
        ch = getchar();
    }
    return x * f;
}

// Uses a large buffer to minimize fwrite() system calls.
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

inline void write_int(long long x) {
    if (x == 0) {
        write_char('0');
        return;
    }
    if (x < 0) {
        write_char('-');
        x = -x;
    }
    static char temp[20]; // Sufficient for a 64-bit integer
    int pos = 0;
    while (x) {
        temp[pos++] = x % 10 + '0';
        x /= 10;
    }
    while (pos--) {
        write_char(temp[pos]);
    }
}

// --- Main Function ---
int main(int argc, char *argv[]) {
    // 1. Argument and file validation
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    std::thread hip_init_thread([]{
        hip_warmup_once();
    });

    // 2. Redirect standard input from file for fast reading
    if (freopen(argv[1], "r", stdin) == NULL) {
        std::cerr << "fileopen error: " << argv[1] << std::endl;
        return 1;
    }

    // 3. Read N using the fast I/O function
    int N = read_int();
    if (N <= 0) {
        write_flush();
        return 0;
    }

    // 4. Allocate pinned host memory for both input and output
    // This is the key optimization for data transfer speed.
    int *h_input_pinned, *h_output_pinned;
    size_t byte_size = (size_t)N * sizeof(int);

    HIP_CHECK(hipHostMalloc(&h_input_pinned, byte_size));
    HIP_CHECK(hipHostMalloc(&h_output_pinned, byte_size));

    // 5. Read the input data directly into the pinned memory
    for (int i = 0; i < N; ++i) {
        h_input_pinned[i] = read_int();
    }
        // 等 HIP 初始化线程收尾（如果它比 I/O 慢，就等待；如果它更快，则已隐藏）
    hip_init_thread.join();

    // 6. Call the GPU solver
    solve(h_input_pinned, h_output_pinned, N);
    // solve(h_input_pinned, h_output_pinned, N);
    
    // 7. Write the output from pinned memory using the fast I/O function
    for (int i = 0; i < N; ++i) {
        write_int(h_output_pinned[i]);
        if (i < N - 1) {
            write_char(' ');
        }
    }
    write_char('\n');
    write_flush(); // Ensure all buffered output is written

    // 8. Free the pinned memory
    HIP_CHECK(hipHostFree(h_input_pinned));
    HIP_CHECK(hipHostFree(h_output_pinned));

    return 0;
}


// #include "main.h"
// #define HIP_CHECK(command) { \
//     hipError_t status = command; \
//     if (status != hipSuccess) { \
//         std::cerr << "HIP Error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
//         exit(EXIT_FAILURE); \
//     } \
// }

// int main(int argc, char *argv[])
// {
//     if (argc != 2)
//     {
//         std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
//         return 1;
//     }

//     std::ifstream input_file;
//     std::string filename = argv[1];

//     input_file.open(filename);
//     if (!input_file.is_open())
//     {
//         std::cerr << "fileopen error " << filename << std::endl;
//         return 1;
//     }

//     int N;
//     input_file >> N;

//     std::vector<int> input(N), output(N);
//     for (int i = 0; i < N; ++i)
//         input_file >> input[i];
//     input_file.close();

//     // --- Pinned Memory Optimization ---
//     int *h_input_pinned, *h_output_pinned;
//     size_t byte_size = N * sizeof(int);

//     // 2. Allocate pinned host memory
//     // std::cout << "Allocating pinned memory..." << std::endl;
//     HIP_CHECK(hipHostMalloc(&h_input_pinned, byte_size));
//     HIP_CHECK(hipHostMalloc(&h_output_pinned, byte_size));
//     memcpy(h_input_pinned, input.data(), byte_size);

//     solve(h_input_pinned, h_output_pinned, N);
//     memcpy(output.data(), h_output_pinned, byte_size);

//     for (int i = 0; i < N; ++i)
//         std::cout << output[i] << " ";
//         // printf("%d ", output[i]);
//     std::cout << std::endl;
//     // std::cout << std::endl;
// }
// #include "main.h"
// #include <chrono>

// int main(int argc, char *argv[])
// {
//     if (argc != 2)
//     {
//         std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
//         return 1;
//     }

//     std::ifstream input_file;
//     std::string filename = argv[1];

//     input_file.open(filename);
//     if (!input_file.is_open())
//     {
//         std::cerr << "fileopen error " << filename << std::endl;
//         return 1;
//     }

//     int N;
//     input_file >> N;

//     std::vector<int> input(N), output(N);
//     for (int i = 0; i < N; ++i)
//         input_file >> input[i];
//     input_file.close();

//     // auto start = std::chrono::high_resolution_clock::now();
//     solve(input.data(), output.data(), N);
//     // auto end = std::chrono::high_resolution_clock::now();
//     // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     // std::cout << "Kernel exteral time: " << duration.count() / 1000.0 << "ms" << std::endl;
//     // solve(input.data(), output.data(), N);

//     for (int i = 0; i < N; ++i)
//         std::cout << output[i] << " ";
//     // printf("%d ", output[i]);
//     std::cout << std::endl;
//     // std::cout << std::endl;
// }
