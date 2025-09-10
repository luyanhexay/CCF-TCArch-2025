#ifndef MAIN_H
#define MAIN_H

#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

static constexpr int INF = 1073741823; // 2^30 - 1

extern "C" void solve(int *dist, int V);

#endif