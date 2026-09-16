#include "../include/utils.h"
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cublas_v2.h>

#define NUM_RUNS 10

#define CUDA_CHECK(func)                                                     	   \
	do {                                                                           \
		cudaError_t status = (func);                                               \
		if (status != cudaSuccess) {                                               \
			printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,   \
				cudaGetErrorString(status), status);                               \
			exit(EXIT_FAILURE);                                                    \
		}                                                                          \
	} while (0)

#define CHECK(name) \
	float *d_Aref_ ## name, *d_Bref_ ## name, *d_Cref_ ## name; \
	std::cerr << "checking " << #name << std::endl; \
	CUDA_CHECK(cudaMalloc(&d_Aref_ ## name, Ref::M * Ref::K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Bref_ ## name, Ref::K * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_Cref_ ## name, Ref::M * Ref::N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_Aref_ ## name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_Bref_ ## name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	float* d_Cref_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < Ref::M; i++) { \
		for (int j = 0; j < Ref::N; j++) { \
			d_Cref_INI_ ## name[i * Ref::N + j] = 0; \
		} \
	} \
	CUDA_CHECK(cudaMemcpy(d_Cref_ ## name, d_Cref_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
	name(d_Aref_ ## name, d_Bref_ ## name, d_Cref_ ## name, Ref::M, Ref::N, Ref::K); \
	cudaError_t err_c_ ## name = cudaGetLastError(); \
	if (err_c_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_ ## name) << std::endl; \
	} \
	CUDA_CHECK(cudaMemcpy(refC, d_Cref_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
	if (!ref.checkRef(refC)){ \
		std::cerr << "check ref failed!" << std::endl; \
	};

#define TIME(name) \
	float *d_A_ ## name, *d_B_ ## name, *d_C_ ## name; \
	CUDA_CHECK(cudaMalloc(&d_A_ ## name, M * K * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_B_ ## name, K * N * sizeof(float))); \
	CUDA_CHECK(cudaMalloc(&d_C_ ## name, M * N * sizeof(float))); \
	CUDA_CHECK(cudaMemcpy(d_A_ ## name, A, M * K * sizeof(float), cudaMemcpyHostToDevice)); \
	CUDA_CHECK(cudaMemcpy(d_B_ ## name, B, K * N * sizeof(float), cudaMemcpyHostToDevice)); \
	cudaEvent_t start_ ## name, end_ ## name; \
	cudaEventCreate(&start_ ## name); \
	cudaEventCreate(&end_ ## name); \
	float* d_C_INI_ ## name = new float[M * N](); \
	for (int i = 0; i < M; i++) { \
		for (int j = 0; j < N; j++) { \
			d_C_INI_ ## name[i * N + j] = 0; \
		} \
	} \
	for (int i = 0; i < 2; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
	} \
	cudaError_t err_t_ ## name = cudaGetLastError(); \
	if (err_t_ ## name != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_ ## name) << std::endl; \
	} \
	float milliseconds_ ## name = 0; \
	for (int i = 0; i < NUM_RUNS; i++) \
	{ \
		CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
		cudaDeviceSynchronize(); \
		cudaEventRecord(start_ ## name); \
		name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
		cudaEventRecord(end_ ## name); \
		cudaEventSynchronize(end_ ## name); \
		float milliseconds_ ## i = 0; \
		cudaEventElapsedTime(&milliseconds_ ## i, start_ ## name, end_ ## name); \
		milliseconds_ ## name += milliseconds_ ## i; \
	} \
	cudaMemcpy(C, d_C_ ## name, M * N * sizeof(float), cudaMemcpyDeviceToHost); \
	std::cout << "Time taken for GEMM (GPU, " << #name <<"): " << milliseconds_ ## name / (float)NUM_RUNS << "ms" << std::endl; \
	cudaFree(d_A_ ## name); \
	cudaFree(d_B_ ## name); \
	cudaFree(d_C_ ## name);

__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C, int M, int N, int K) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < K; k++) {
					C[i * N + j]  += A[i * K + k]  * B[k * N + j];
				}
			}
		}
    }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 blockSize(1);
	dim3 gridSize(1);
	gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// The scafolding for optimized GEMM implementations
#define BLOCK_I 16
#define BLOCK_J 16

// each thread owns one output
__global__ void gemm_gpu_o1_kernel(float* A, float* B, float *C, int M, int N, int K) {
	int i = blockIdx.x * BLOCK_I + threadIdx.x;
	int j = blockIdx.y * BLOCK_J + threadIdx.y;
	
    if (i < M && j < N) {
    	float sum = 0;
    	for (int k = 0; k < K; k++) {
    		sum += A[i * K + k] * B[k * N + j];
    	}
    	C[i * N + j] = sum;
    }
}

void gemm_gpu_o1(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 gridSize(cuda::ceil_div(M, BLOCK_I), cuda::ceil_div(N, BLOCK_J));
	dim3 blockSize(BLOCK_I, BLOCK_J);
	gemm_gpu_o1_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

#define TILE 16
// each thread owns output i, j
// block is (TILE, TILE)
__global__ void gemm_gpu_o2_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float sa[TILE][TILE];
	__shared__ float sb[TILE][TILE];
	int i = blockIdx.y * TILE + threadIdx.y;
	int j = blockIdx.x * TILE + threadIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int blocks = cuda::ceil_div(K, TILE);
	float sum = 0;
	for (int block = 0; block < blocks; block++) {
		// load shared memory
		if (i < M && block * TILE + tx < K)
			sa[ty][tx] = A[i * K + block * TILE + tx];
		else 
			sa[ty][tx] = 0;
		if (j < N && block * TILE + ty < K)
			sb[ty][tx] = B[(block * TILE + ty) * N + j];
		else 
			sb[ty][tx] = 0;
		__syncthreads();
		// accumulate
		for (int k = 0; k < TILE; k++){
			sum += sa[ty][k] * sb[k][tx];
		}
		__syncthreads();
	}
	if (i < M && j < N) {
		C[i * N + j] = sum;
	}
	
}
void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K)
{
	// Init block and grid size
	dim3 gridSize(cuda::ceil_div(M, TILE), cuda::ceil_div(N, TILE));
	dim3 blockSize(TILE, TILE);
	gemm_gpu_o2_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

// feel like doing register tiling right now
// dont really know why
#define BTILE 64
#define KTILE 8
#define TTILE 8
__global__ void gemm_gpu_o3_kernel(float* A, float* B, float *C, int M, int N, int K) {
	__shared__ float sa[BTILE][KTILE];
    __shared__ float sb[KTILE][BTILE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BTILE / TTILE) + tx;
    int num_threads = (BTILE / TTILE) * (BTILE / TTILE);
    int block_j = blockIdx.y * BTILE;
    int block_i = blockIdx.x * BTILE;
    int thread_j = ty * TTILE;
    int thread_i = tx * TTILE;
    int global_j_base = block_j + thread_j;
    int global_i_base = block_i + thread_i;

    float sum[TTILE][TTILE] = {0};
    float reg_a[TTILE];
    float reg_b[TTILE];

    int blocks = cuda::ceil_div(K, KTILE);

    for (int block = 0; block < blocks; block++) {
        int k_offset = block * KTILE;

		// load sa
        #pragma unroll
        for (int load_idx = tid; load_idx < BTILE * KTILE; load_idx += num_threads) {
             int lj = load_idx / KTILE;
             int lk = load_idx % KTILE;
             int gj = block_j + lj;
             int gk = k_offset + lk;

			if (gj < M && gk < K)
            	sa[lj][lk] = A[gj * K + gk];
            else
            	sa[lj][lk] = 0;
        }

		// load sb
        #pragma unroll
        for (int load_idx = tid; load_idx < KTILE * BTILE; load_idx += num_threads) {
             int lk = load_idx / BTILE;
             int li = load_idx % BTILE;
             int gk = k_offset + lk;
             int gi = block_i + li;

			if (gk < K && gi < N) 
            	sb[lk][li] = B[gk * N + gi];
            else
            	sb[lk][li] = 0;
        }
        __syncthreads();

		// register tiled compute
        #pragma unroll
        for (int dot_idx = 0; dot_idx < KTILE; dot_idx++) {
        	// load register a
            #pragma unroll
            for (int j = 0; j < TTILE; j++) {
                reg_a[j] = sa[thread_j + j][dot_idx];
            }
			// load register b
            #pragma unroll
            for (int i = 0; i < TTILE; i++) {
                reg_b[i] = sb[dot_idx][thread_i + i];
            }
			// accumulate
            #pragma unroll
            for (int j = 0; j < TTILE; j++) {
                #pragma unroll
                for (int i = 0; i < TTILE; i++) {
                    sum[j][i] += reg_a[j] * reg_b[i];
                }
            }
        }
        __syncthreads();
    }

	// writeback
    #pragma unroll
    for (int j = 0; j < TTILE; j++) {
         int gj = global_j_base + j;
        if (gj < M) {
            #pragma unroll
            for (int i = 0; i < TTILE; i++) {
                 int gi = global_i_base + i;
                if (gi < N) {
                    C[gj * N + gi] = sum[j][i];
                }
            }
        }
    }
}
void gemm_gpu_o3(float* A, float* B, float* C, int M, int N, int K)
{
	dim3 blockSize(BTILE / TTILE, BTILE / TTILE);
    dim3 gridSize(cuda::ceil_div(N, BTILE), cuda::ceil_div(M, BTILE));
    gemm_gpu_o3_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void gemm_gpu_cublas(float* A, float* B, float* C, int M, int N, int K)
{
	cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1;
    float beta = 1;
    cublasSgemm(
    	handle,
    	CUBLAS_OP_N,
    	CUBLAS_OP_N,
    	N, M, K,
    	&alpha,
    	B, N,
    	A, K,
    	&beta,
    	C, N
    );
}


int main(int argc, char* argv[]) {
	if (argc < 3) {
		std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
		return 1;
	}

	int run_o0 = 1;
	if (argc >= 5) run_o0 = atoi(argv[4]);
	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	// int runs = atoi(argv[3]);
	float* A = new float[M * K]();
	float* B = new float[K * N]();
	float* C = new float[M * N]();

	fillRandom(A, M * K);
	fillRandom(B, K * N);

	/// GPU Implementation
        // Check if implementation is correct
	auto ref = Ref();
	float* refC = new float[Ref::M * Ref::N]();
	if (run_o0 > 0) {
 		CHECK(gemm_gpu_o0)
 	}
 	if (run_o0 != -1){
		CHECK(gemm_gpu_o1)
		CHECK(gemm_gpu_o2)
	}
	CHECK(gemm_gpu_o3)
	CHECK(gemm_gpu_cublas)

	// Actual run
	if (run_o0 > 0) {
 		TIME(gemm_gpu_o0)
 	}
 	if (run_o0 != -1){
		TIME(gemm_gpu_o1)
		TIME(gemm_gpu_o2)
	}
	TIME(gemm_gpu_o3)
	TIME(gemm_gpu_cublas)

	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	delete[] A;
	delete[] B;
	delete[] C;

	return 0;
}
