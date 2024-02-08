#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048
#define BLOCK_SIZE 16 

void checkCUDAError(const char*);
void random_matrix(int *a, int n);
void matrixAddCPU(int *a, int *b, int *c, int max);
int validate(int *c, int *c_ref, int max);

__global__ void matrixAdd(int *a, int *b, int *c, int max) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * max + col;
    if (row < max && col < max) {
        c[index] = a[index] + b[index];
    }
}

void matrixAddCPU(int *a, int *b, int *c, int max) {
    for (int i = 0; i < max; i++) {
        for (int j = 0; j < max; j++) {
            int index = i * max + j;
            c[index] = a[index] + b[index];
        }
    }
}

int validate(int *c, int *c_ref, int max) {
    int e = 0;
    for (int i = 0; i < max * max; i++) {
        if (c[i] != c_ref[i]) {
            printf("Error at %d: GOT {%d} in GPU and GOT {%d} in CPU\n", i, c[i], c_ref[i]);
            e++;
        }
    }
    return e;
}

int main(void) {
    int *a, *b, *c, *c_ref;     
    int *d_a, *d_b, *d_c;                   
    int errors;
    unsigned int size = N * N * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    checkCUDAError("CUDA malloc");

    a = (int *)malloc(size); random_matrix(a, N);
    b = (int *)malloc(size); random_matrix(b, N);
    c = (int *)malloc(size);
    c_ref = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy");

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();
    checkCUDAError("CUDA kernel");

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy");

    matrixAddCPU(a, b, c_ref, N);
    errors = validate(c, c_ref, N);
    printf("Errors: %d\n", errors);

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
        a[i] = rand();
    }
}

