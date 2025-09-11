#include "main.h"
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>

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