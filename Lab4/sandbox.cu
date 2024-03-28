#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 4194304
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);


// task 1.1
__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

__global__ void vectorAdd(int max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main(void) {
    int *a, *b, *c;			// host copies of a, b, c
    int errors;
    unsigned int size = N * sizeof(int);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a);
    b = (int *)malloc(size); random_ints(b);
    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpyToSymbol(d_a, a, size);
    cudaMemcpyToSymbol(d_b, b, size);
    checkCUDAError("CUDA memcpy to symbol");

    // task 1.2 Record timings
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch add() kernel on GPU
    vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(N);
    checkCUDAError("CUDA kernel");

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to host
    cudaMemcpyFromSymbol(c,d_c,size);
    checkCUDAError("CUDA memcpy");

    printf("Kernel Execution Time: %fms\n", ms);

    // task 1.3
    cudaDeviceProp d_prop;
    int deviceId;
    double memoryClockRate, memoryBusWidth, theoreticalBW;

    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&d_prop,deviceId);

    memoryClockRate = d_prop.memoryClockRate * 1e-6;
    memoryBusWidth = d_prop.memoryBusWidth;
    theoreticalBW = ((memoryClockRate * memoryBusWidth) * 2)/8;

    printf("Theoretical Memory Bandwidth: %.2fGB/s\n",theoreticalBW);

    // task 1.4
    double r_bytes, w_bytes, measuredBW;
    r_bytes = N * 8;
    w_bytes = N * 4;
    measuredBW = ((r_bytes + w_bytes)/(ms/1000))/1e9 ;

    printf("Measured Memory Bandwidth: %.2fGB/s\n",measuredBW);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(a); free(b); free(c);
    checkCUDAError("CUDA cleanup");

    return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void random_ints(int *a)
{
    for (unsigned int i = 0; i < N; i++){
        a[i] = rand();
    }
}