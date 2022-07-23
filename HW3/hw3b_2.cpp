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
#define ROUND_MAX 4

int n, m;
int dist[V*V];
size_t pitch;
cudaDeviceProp prop;
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

__global__ void p1_cal_kernel(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int n, int* Dist_gpu, int pitch_int) {

	int b_i = block_start_x ;
	int b_j = block_start_y ;

	//int inner_round = (B*B-1)/blockDim.x + 1;

	extern __shared__ int shared_mem[];
	int global_i[ROUND_MAX];
	int global_j[ROUND_MAX];
	int inner_i[ROUND_MAX];
	int inner_j[ROUND_MAX];

	#pragma unroll
	for(int r=0; r<4; r++){
		inner_i[r] = threadIdx.y + 16 * r;
		inner_j[r] = threadIdx.x;
		//if(inner_i[r]>=B) continue;
		global_i[r] = b_i * B + inner_i[r];
		global_j[r] = b_j * B + inner_j[r];
		if (!((global_i[r]>=n) | (global_j[r]>=n)))
			shared_mem[inner_i[r]*B + inner_j[r]] = Dist_gpu[global_i[r]*pitch_int + global_j[r]];
	}

	for (int k = 0; k <  B && (k+Round*B) < n; ++k) {
		__syncthreads();

		#pragma unroll
		for(int r=0; r<4; r++){
			//if(inner_i[r]>=B) continue;
			if ((global_i[r]>=n) | (global_j[r]>=n)) continue ;

			if (shared_mem[inner_i[r]*B+inner_j[r]] > shared_mem[inner_i[r]*B+k] + shared_mem[k*B+inner_j[r]]) {
				shared_mem[inner_i[r]*B+inner_j[r]] = shared_mem[inner_i[r]*B+k] + shared_mem[k*B+inner_j[r]];
			}
		}

	}

	#pragma unroll
	for(int r=0; r<4; r++){
		//if(inner_i[r]>=B) continue;
		if (!((global_i[r]>=n) | (global_j[r]>=n)))
			Dist_gpu[global_i[r]*pitch_int + global_j[r]] = shared_mem[inner_i[r]*B + inner_j[r]];
	}

}


extern __shared__ int shared_mem[];
__global__ void p2_cal_kernel(int B, int Round, int n, int* Dist_gpu, int pitch_int) {

	int b_i, b_j;
	if(blockIdx.y==0){
		b_i = Round;
		b_j = blockIdx.x + (blockIdx.x>=Round);
	}
	else{
		b_i = blockIdx.x + (blockIdx.x>=Round);
		b_j = Round;
	}

	//int inner_round = (B*B-1)/blockDim.x + 1;


	int global_i[ROUND_MAX];
	int global_j[ROUND_MAX];
	int inner_i[ROUND_MAX];
	int inner_j[ROUND_MAX];

	#pragma unroll
	for(int r=0; r<4; r++){
		inner_i[r] = threadIdx.y + 16 * r;
		inner_j[r] = threadIdx.x;
		//if(inner_i[r]>=B) continue;
		global_i[r] = b_i * B + inner_i[r];
		global_j[r] = b_j * B + inner_j[r];
		int global_pivot_i = Round * B + inner_i[r];
		int global_pivot_j = Round * B + inner_j[r];
		if (!((global_i[r]>=n) | (global_j[r]>=n)))
			shared_mem[inner_i[r]*B + inner_j[r]] = Dist_gpu[global_i[r]*pitch_int + global_j[r]];
		if (!((global_pivot_i>=n) | (global_pivot_j>=n)))
			shared_mem[inner_i[r]*B + inner_j[r] + B*B] = Dist_gpu[global_pivot_i*pitch_int + global_pivot_j];
	}


	for (int k = 0; k <  B && (k+Round*B) < n; ++k) {
		__syncthreads();

		#pragma unroll
		for(int r=0; r<4; r++){
			//if(inner_i[r]>=B) continue;
			if ((global_i[r]>=n) | (global_j[r]>=n)) continue ;

			//if ((Dist_gpu[i*n+k] + Dist_gpu[k*n+j])==73) printf("%d, %d, %d, %d\n", i, j, k, n);
			if (shared_mem[inner_i[r]*B+inner_j[r]] > shared_mem[inner_i[r]*B+k + !blockIdx.y*B*B] + shared_mem[k*B+inner_j[r] + blockIdx.y*B*B]) {
				shared_mem[inner_i[r]*B+inner_j[r]] = shared_mem[inner_i[r]*B+k + !blockIdx.y*B*B] + shared_mem[k*B+inner_j[r] + blockIdx.y*B*B];
			}
		}
	}
	#pragma unroll
	for(int r=0; r<4; r++){
		//if(inner_i[r]>=B) continue;
		if (!((global_i[r]>=n) | (global_j[r]>=n)))
			Dist_gpu[global_i[r]*pitch_int + global_j[r]] = shared_mem[inner_i[r]*B + inner_j[r]];

	}


}

__global__ void p3_cal_kernel(int B, int Round, int n, int* Dist_gpu, int pitch_int) {

	int b_i = blockIdx.y + (blockIdx.y>=Round);
	int b_j = blockIdx.x + (blockIdx.x>=Round);

	__shared__ int shared_mem[8192];
	//int inner_round = (B*B-1)/blockDim.x + 1;

	int global_i[ROUND_MAX];
	int global_j[ROUND_MAX];
	int inner_i[ROUND_MAX];
	int inner_j[ROUND_MAX];
	int my_dist[ROUND_MAX];

	#pragma unroll
	for(int r=0; r<4; r++){
		//if(inner_i[r]>=B) continue;
		inner_i[r] = threadIdx.y + 16 * r;
		inner_j[r] = threadIdx.x;
		global_i[r] = b_i * B + inner_i[r];
		global_j[r] = b_j * B + inner_j[r];
		int row_pivot_i = global_i[r];
		int row_pivot_j = Round * B + inner_j[r];
		int col_pivot_i = Round * B + inner_i[r];
		int col_pivot_j = global_j[r];

		my_dist[r] = Dist_gpu[global_i[r]*pitch_int + global_j[r]];
		shared_mem[inner_i[r]*B + inner_j[r] ] = Dist_gpu[row_pivot_i*pitch_int + row_pivot_j];
		shared_mem[inner_i[r]*B + inner_j[r] + B*B] = Dist_gpu[col_pivot_i*pitch_int + col_pivot_j];

	}

	__syncthreads();
	for (int k = 0; k <  B && (k+Round*B) < n; ++k) {
		#pragma unroll
		for(int r=0; r<4; r++){
			int tmp = shared_mem[inner_i[r]*B+k ] + shared_mem[k*B+inner_j[r] +B*B];
			if (my_dist[r] > tmp) {
				my_dist[r] = tmp;
			}
		}
	}

	#pragma unroll
	for(int r=0; r<4; r++){
		Dist_gpu[global_i[r]*pitch_int + global_j[r]] = my_dist[r];

	}

}

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B) {
    int round = ceil(n, B);
    int num_thread = (B*B>prop.maxThreadsPerBlock)? prop.maxThreadsPerBlock: B*B;
    int *device_dist;
    cudaMallocPitch((void**)&device_dist, &pitch,n*sizeof(int), n+64);
	  int pitch_int = pitch / sizeof(int);
	  cudaMemcpy2D(device_dist, pitch, dist, n*sizeof(int), n*sizeof(int), n, cudaMemcpyHostToDevice);
    dim3 grid2(round-1, 2);
	  dim3 grid3(round-1, round-1);
	  dim3 block(B, num_thread/B);
    for (int r = 0; r < round; r++) {
      p1_cal_kernel<<< 1, block, B*B*sizeof(int)>>>(B, r,	r,	r,	1,	1, n, device_dist, pitch_int);

      p2_cal_kernel<<< grid2, block, 2*B*B*sizeof(int) >>>(B, r, n, device_dist, pitch_int);

      p3_cal_kernel<<< grid3, block>>>(B, r, n, device_dist, pitch_int);
    }
    cudaMemcpy2D(dist, n*sizeof(int), device_dist, pitch, n*sizeof(int), n, cudaMemcpyDeviceToHost);
    // cudaFree(device_dist);
}

int main(int argc, char* argv[]) {
    // dist = (int*)malloc(sizeof(int)*V*V);
    input(argv[1]);
    cudaGetDeviceProperties(&prop, 0);

    int B = 32;
    block_FW(B);

    output(argv[2]);
    // cudaFree(r_dist);
    return 0;
}
