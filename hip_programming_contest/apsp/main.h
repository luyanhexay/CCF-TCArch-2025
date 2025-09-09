#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <fstream>

static constexpr int INF = 1073741823; // 2^30 - 1

extern "C" void solve(int* dist, int V);

#endif 