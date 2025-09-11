#include "main.h" // 假设你的 solve 函数声明在此
#include <cstdio>
#include <cctype>
#include <vector>

// --- 快速读入函数 ---
inline int read_int() {
    int x = 0, f = 1;
    char ch = getchar();
    while (!isdigit(ch)) {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (isdigit(ch)) {
        x = x * 10 + (ch - '0');
        ch = getchar();
    }
    return x * f;
}

// --- 快速输出函数（带缓冲区）---
const int OUT_BUFFER_SIZE = 1 << 20; // 1MB 缓冲区
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

inline void write_int(int x) {
    if (x == 0) {
        write_char('0');
        return;
    }
    if (x < 0) {
        write_char('-');
        x = -x;
    }
    static char temp[12];
    int pos = 0;
    while (x) {
        temp[pos++] = x % 10 + '0';
        x /= 10;
    }
    while (pos--) {
        write_char(temp[pos]);
    }
}

// --- 主函数 ---
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // 重新定向标准输入，从文件读取
    // freopen 会关闭旧的流，并用新的文件流打开
    if (freopen(argv[1], "r", stdin) == NULL) {
        std::cerr << "fileopen error: " << argv[1] << std::endl;
        return 1;
    }

    // 关闭 C++ 流与 C 流的同步，并解绑 cin/cout
    // 虽然我们不使用它们，但这是好习惯，避免潜在问题
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    int V = read_int();
    int E = read_int();

    if (V <= 0) {
        write_flush(); // 确保输出缓冲区被清空
        return 0;
    }

    std::vector<int> dist((size_t)V * (size_t)V, INF);
    for (int i = 0; i < V; ++i) {
        dist[(size_t)i * (size_t)V + i] = 0;
    }

    for (int e = 0; e < E; ++e) {
        int u = read_int();
        int v = read_int();
        int w = read_int();
        if (u >= 0 && u < V && v >= 0 && v < V) {
            size_t idx = (size_t)u * (size_t)V + v;
            if (w < dist[idx]) {
                dist[idx] = w;
            }
        }
    }
    // freopen 已经将文件关闭，不需要再调用 close

    // 调用 GPU 求解器
    solve(dist.data(), V);

    // 使用快速输出
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            write_int(dist[(size_t)i * (size_t)V + j]);
            if (j + 1 < V) {
                write_char(' ');
            }
        }
        write_char('\n');
    }

    write_flush(); // 务必在程序结束前调用，将缓冲区内容写入 stdout

    return 0;
}