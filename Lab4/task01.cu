#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 4194304
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__device__ int d_a[N];
__device__ int d_b[N];
__device__ int d_c[N];

//__global__ void vectorAdd(int *d_a, int *d_b, int *d_c, int max) {
__global__ void vectorAdd(int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	d_c[i] = d_a[i] + d_b[i];
}

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
//	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
//	cudaMalloc((void **)&d_a, size);
//	cudaMalloc((void **)&d_b, size);
//	cudaMalloc((void **)&d_c, size);
//	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
//	c_ref = (int *)malloc(size);

	// Copy inputs to device
//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_a, a, size);
    cudaMemcpyToSymbol(d_b, b, size);
//	checkCUDAError("CUDA memcpy");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
	vectorAdd << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(N);
	checkCUDAError("CUDA kernel");

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Start: %f, Stop: %f, Milliseconds: %f\n", start, stop, milliseconds);

	// Copy result back to host
//	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
//	checkCUDAError("CUDA memcpy");

    cudaDeviceProp device_prop; int deviceId;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&device_prop, deviceId);
    float memoryClockRate, memoryBusWidth, theoreticalBW;

    memoryClockRate = device_prop.memoryClockRate * 1e-6;
    memoryBusWidth = device_prop.memoryBusWidth;
    theoreticalBW = (((float) memoryClockRate * memoryBusWidth) * 2) / 8;

    float read_bytes, write_bytes, measuredBW;
    read_bytes = N * 8;
    write_bytes = N * 4;
    measuredBW = ((read_bytes + write_bytes) / (milliseconds / 1000)) / 1e9;

    printf("Took %.2f milliseconds\n", milliseconds);
    printf("Our theoretical memory bandwidth: %.2f GB per second\n", theoreticalBW);
    printf("Our memory bandwidth: %.2f GB per second\n", measuredBW);

    cudaEventDestroy(start); cudaEventDestroy(stop);
	// Cleanup

	free(a); free(b); free(c);
//	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
//	checkCUDAError("CUDA cleanup");

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
