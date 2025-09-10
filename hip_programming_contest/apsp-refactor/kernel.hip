#include "main.h"
#include "blocked_floyd_warshall.h"

extern "C" void solve(int* dist, int V) {
    if (V <= 0) return;
    
    // 使用默认块大小16，可以根据需要调整
    int block_size = 16;
    if (V < block_size) block_size = V;
    
    t_error err;
    int result = blocked_floyd_warshall(dist, V, block_size, dist, &err);
    
    if (result != 0) {
        std::cerr << "Error in blocked_floyd_warshall: " << err.err_msg << std::endl;
    }
}
