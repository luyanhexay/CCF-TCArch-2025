#include "blocked_floyd_warshall.h"
#include "errors.h"
#include "main.h"

void solve(int *dist, int V)
{
    t_error err;
    blocked_floyd_warshall(dist, V, 16, dist, &err);
}