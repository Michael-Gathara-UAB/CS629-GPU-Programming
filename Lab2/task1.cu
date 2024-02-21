#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 8
#define K 8
#define M 8
#define BLOCK_SIZE 16 

void checkCUDAError(const char*);
void random_matrix(float *a, int n);

void matrixMultCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

__global__ 
void matrixMultGPUKernel(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
           tmp += a[row * n + i] * b[i * n + col];
        }
    }
    c[row * n + col] = tmp;
}

int validate(float *c, float *c_ref, int N) {
    int e = 0;
    for (int i = 0; i < N * N; i++) {
        if (c[i] != c_ref[i]) {
            printf("Error at %d: GOT {%d} in GPU and GOT {%d} in CPU\n", i, c[i], c_ref[i]);
            e++;
        }
    }
    return e;
}

int main(void) {
    float *a, *b, *c, *c_ref;     
    float *d_a, *d_b, *d_c;                   
    int errors;
    unsigned int size = N * N :wq

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    checkCUDAError("CUDA malloc");

    constexpr A_size = M * K;
    constexpr B_size = K * N;
    constexpr C_size = M * N;

    a = (float *)malloc(A_size); random_matrix(a, N);
    b = (float *)malloc(B_size); random_matrix(b, N);
    c = (float *)malloc(C_size);
    c_ref = (float *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy");

   	 
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    checkCUDAError("CUDA kernel");

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy");

    errors = validate(c, c_ref);
    free(a); free(b); free(c); free(c_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    checkCUDAError("CUDA cleanup");

    return 0;
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void random_matrix(int *a, int n) {
    for (unsigned int i = 0; i < n * n; i++) {
        a[i] = rand() % 10001 / 10000.0;
    }
}

