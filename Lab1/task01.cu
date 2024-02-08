#include <stdio.h>
#include <stdlib.h>
#include <typeinfo>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1024
#define A_INV 111
#define B 27
#define M 128 

void checkCUDAError(const char*);
void read_encrypted_file(int*);

__device__ 
int modulo(int a, int b) {
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

/*
  The encryption function: E(x) = (Ax + B) mod M
  -> A and B are keys of the cipher
*/
__global__ 
void affine_decrypt(int *d_input, int *d_output) {
    int idx = threadIdx.x;
    if (idx < N) {
        d_output[idx] = modulo(A_INV * (d_input[idx] - B), M);
    }
}

__global__
void affine_decrypt_multiblock(int *d_input, int *d_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_output[idx] = modulo(A_INV * (d_input[idx] - B), M);
    }
}

int main(int argc, char *argv[]) {
	int *h_input, *h_output;
	int *d_input, *d_output;
	unsigned int size;
	int i;

	size = N * sizeof(int);
	h_input = (int *)malloc(size);
	h_output = (int *)malloc(size);
  cudaMalloc((void **)&d_input, size);
  cudaMalloc((void **)&d_output, size);
	checkCUDAError("Memory allocation");
	read_encrypted_file(h_input);
  cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	checkCUDAError("Input transfer to device");
  dim3 blocksPerGrid(8, 1, 1);
  dim3 threadsPerBlock(128, 1, 1);
  affine_decrypt_multiblock<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel execution");
  cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
	checkCUDAError("Result transfer to host");

	for (i = 0; i < N; i++) {
		printf("%c", (char)h_output[i]);
	}
	printf("\n");

	/* Exercise 1.7: free device memory */
	//cudaFree(???);
	//cudaFree(???);
  cudaFree(d_input);
  cudaFree(d_output);
	checkCUDAError("Free memory");

	/* free host buffers */
	free(h_input);
	free(h_output);

	return 0;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void read_encrypted_file(int* input) {
	FILE *f = NULL;
  	f = fopen("encrypted01.bin", "rb");
	if (f == NULL){
		fprintf(stderr, "Error: Could not find encrypted01.bin file \n");
		exit(1);
	}
	//read encrypted data
	fread(input, sizeof(unsigned int), N, f);
	fclose(f);
}
