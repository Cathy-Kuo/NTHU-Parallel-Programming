#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <assert.h>

const int INF = ((1 << 30) - 1);
const int V = 12010;
void input(char* inFileName);
void output(char* outFileName);
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

int n, m;
int dist[V*V];
// __device__ static int devDist[V][V];

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                dist[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        dist[pair[0]*n+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    fwrite(dist, sizeof(int), n*n, outfile);
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }
__device__ int Min(int a, int b) { return a < b ? a : b; }

__global__
void SharedMemoryFloydWarshall(int* device_dist, int k, int n) {
    __shared__ int dist_i_k[TILE_HEIGHT];
    __shared__ int dist_k_j[TILE_WIDTH];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < n && j < n) {
        int dist_i_j = device_dist[i*n + j];
        if (i % TILE_HEIGHT == 0) {
            dist_k_j[j % TILE_WIDTH] = device_dist[k*n + j];
        }
        if (j % TILE_WIDTH == 0) {
            dist_i_k[i % TILE_HEIGHT] = device_dist[i*n + k];
        }
        __syncthreads();
        if (dist_i_k[i % TILE_HEIGHT] != INF && dist_k_j[j % TILE_WIDTH] != INF) {
            int new_dist = dist_i_k[i % TILE_HEIGHT] + dist_k_j[j % TILE_WIDTH];
            if (new_dist<dist_i_j) device_dist[i*n + j] = new_dist;
        }
    }
}

void block_FW(int B, dim3 thread_per_block) {
    int round = n;
    int *device_dist;
    cudaMalloc(&device_dist, sizeof(int) * n * n);
    cudaMemcpy(device_dist, dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    dim3 num_block(ceil(1.0*n/thread_per_block.x),
                   ceil(1.0*n/thread_per_block.y));
    for (int k = 0; k < n; ++k) {
        SharedMemoryFloydWarshall<<<num_block, thread_per_block>>>(device_dist, k, n);
    }
    cudaMemcpy(dist, device_dist, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
    // cudaFree(device_dist);
}

int main(int argc, char* argv[]) {
    // dist = (int*)malloc(sizeof(int)*V*V);
    input(argv[1]);

    int B = 1;
    dim3 thread_per_block(TILE_HEIGHT, TILE_WIDTH);
    block_FW(B, thread_per_block);

    output(argv[2]);
    // cudaFree(r_dist);
    return 0;
}
