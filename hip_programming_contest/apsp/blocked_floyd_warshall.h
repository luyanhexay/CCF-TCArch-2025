#ifndef BLOCKED_FLOYD_WARSHALL_H
#define BLOCKED_FLOYD_WARSHALL_H

// Error handling structure
typedef struct error
{
    int err_code;
    char err_msg[100];
} t_error;

// Error codes
#define WRONG_NUM_OF_NODES_ERR 1
#define WRONG_BLOCK_SIZE_ERR 2

/**
 * @param graph input graph
 * @param n number of graph's vertices
 * @param b block size, (0 < b <= n)
 * @param apsp all pairs shortest paths
 * @param err in case of error during the execution, this will contain some info about it
 * @return 0 if success, non-zero otherwise
 */
int blocked_floyd_warshall(int *graph, int n, int b, int *apsp, t_error *err);

#endif // BLOCKED_FLOYD_WARSHALL_H
