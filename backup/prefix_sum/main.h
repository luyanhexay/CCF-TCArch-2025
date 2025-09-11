#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <thread>
#include <fstream>

#define HIP_CHECK(call) do { \
    hipError_t error = call; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

extern "C" void solve(const int* input, int* output, int N);

#endif 