#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <assert.h>

const int INF = ((1 << 30) - 1);
const int V = 10010;
void input(char* inFileName);
void output(char* outFileName);

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

__global__ void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* device_dist) {
    // int block_end_x = block_start_x + block_height;
    // int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x+blockIdx.x; b_i < block_start_x + block_height; b_i+= gridDim.x) {
        for (int b_j = block_start_y; b_j < block_start_y + block_width; b_j++) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            // for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                // int block_internal_start_x = b_i *B;
                int block_internal_end_x = (b_i + 1) * B;
                // int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int i = b_i+threadIdx.x; i < block_internal_end_x; i+=blockDim.x) {
                    for (int j = b_j; j < block_internal_end_y; j++) {
                        if (device_dist[i*n+Round] + device_dist[Round*n+j] < device_dist[i*n+j]) {
                            device_dist[i*n+j] = device_dist[i*n+Round] + device_dist[Round*n+j];
                            // __syncthreads();
                        }
                    }
                }

            // }
        }
    }
}

void block_FW(int B) {
    int round = n;
    int *device_dist;
    cudaMalloc(&device_dist, sizeof(int) * n * n);
    cudaMemcpy(device_dist, dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    for (int r = 0; r < round; r++) {
        cal<<<96,96>>>(B, r, r, r, 1, 1, n, device_dist);

        /* Phase 2*/
        cal<<<96,96>>>(B, r, r, 0, r, 1, n, device_dist);
        cal<<<96,96>>>(B, r, r, r + 1, round - r - 1, 1, n, device_dist);
        cal<<<96,96>>>(B, r, 0, r, 1, r, n, device_dist);
        cal<<<96,96>>>(B, r, r + 1, r, 1, round - r - 1, n, device_dist);

        /* Phase 3*/
        cal<<<96,96>>>(B, r, 0, 0, r, r, n, device_dist);
        cal<<<96,96>>>(B, r, 0, r + 1, round - r - 1, r, n, device_dist);
        cal<<<96,96>>>(B, r, r + 1, 0, r, round - r - 1, n, device_dist);
        cal<<<96,96>>>(B, r, r + 1, r + 1, round - r - 1, round - r - 1, n, device_dist);
    }
    cudaMemcpy(dist, device_dist, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
    // cudaFree(device_dist);
}

int main(int argc, char* argv[]) {
    // dist = (int*)malloc(sizeof(int)*V*V);
    input(argv[1]);

    int B = 1;
    block_FW(B);

    output(argv[2]);
    // cudaFree(r_dist);
    return 0;
}
