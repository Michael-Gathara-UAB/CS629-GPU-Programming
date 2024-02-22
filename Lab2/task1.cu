/*
Author: Michael Gathara (mikegtr at uab dot edu)*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16  
void random_matrix(int *arr, int n);

__global__ void matrixMultGPUKernel(float *a, float *b, float *c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < m && col < n) {
        float sum = 0.0;
        for (int i = 0; i < k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

void matrixMultCPU(float *a, float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0;
            for (int x = 0; x < k; ++x) {
                sum += a[i * k + x] * b[x * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void random_matrix(float *arr, int n) {
    for (unsigned int i = 0; i < n * n; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int validate(float *c_gpu, float *c_cpu, int m, int n) {
    int errors = 0;
    for (int i = 0; i < m * n; i++) {
        // https://www.oreilly.com/library/view/c-in-a/0596006977/re57.html#:~:text=The%20fabs()%20function%20returns,%2C%20the%20function%20returns%20%2Dx%20.
        // Used the above link for fabs
        if (fabs(c_gpu[i] - c_cpu[i]) > 1) {
            // There might be soem floating point errors so I'm only checking to make sure the whole number section is the same
            errors++;
            printf("Mismatch at %d: GPU = %f, CPU = %f\n", i, c_gpu[i], c_cpu[i]);
        }
    }
    return errors;
}

int main() {
    int lim = 10000;
    unsigned int m = lim, k = lim, n = lim; 
    size_t a_size = m * k * sizeof(float);
    size_t b_size = k * n * sizeof(float);
    size_t c_size = m * n * sizeof(float);

    float *a, *b, *c_gpu, *c_cpu;
    a = (float*)malloc(a_size);
    b = (float*)malloc(b_size);
    c_gpu = (float*)malloc(c_size);
    c_cpu = (float*)malloc(c_size);

    random_matrix(a, m);
    random_matrix(b, k);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_c, c_size);

    cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);

    int blockSizeArray[] = {8, 16, 32};
    for (int blockSizeIndex = 0; blockSizeIndex < 3; blockSizeIndex++) {
        int blockSize = blockSizeArray[blockSizeIndex];
        dim3 threads(blockSize, blockSize);
        dim3 grid((n + blockSize - 1) / blockSize, (m + blockSize - 1) / blockSize);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        matrixMultGPUKernel<<<grid, threads>>>(d_a, d_b, d_c, m, k, n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(c_gpu, d_c, c_size, cudaMemcpyDeviceToHost);
        matrixMultCPU(a, b, c_cpu, m, k, n);

        int errors = validate(c_gpu, c_cpu, m, n);
        printf("Block size %dx%d, Errors: %d\n", blockSize, blockSize, errors);

        float gflops = (m * k * n * 2.0) / 1e9 / (milliseconds / 1000);
        printf("Block size %dx%d, GFLOPS: %f\n", blockSize, blockSize, gflops);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    free(a); free(b); free(c_gpu); free(c_cpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
