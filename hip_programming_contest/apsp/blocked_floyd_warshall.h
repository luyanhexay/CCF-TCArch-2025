#ifndef TYPEDEF_ERROR
#define TYPEDEF_ERROR
#include "errors.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @param graph input graph
 * @param n number of graph's vertices
 * @param b block size, (0 < b <= n)
 * @param apsp all pairs shortest paths
 * @param err in case of error during the execution, this will contain some info about it
 * @return 0 if success, non-zero otherwise
 */
int blocked_floyd_warshall(int* graph, int n, int b, int* apsp, t_error* err);

#ifdef __cplusplus
}
#endif
