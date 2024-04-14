#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#define HISTO_SIZE 256


void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void generate_data(char * data, unsigned int length) {
	for (int i = 0; i < length; ++i) {
		data[i] = (char)rand() % HISTO_SIZE;
	}
}

__global__ void GPU_histogram(char * data, unsigned int length, unsigned int* histo) {
	// Task 1.1 Add your implementation here
}

void CPU_histogram(char * data, unsigned int length, unsigned int* histo) {
	// Task 1.2 Add a CPU implementation for verification

}

/* Host code */
int main(void) {
	unsigned int input_length = 2048;
	char * h_data, * d_data;
	unsigned int * h_histo, * d_histo;
	cudaEvent_t start, stop;
	float ms;

	unsigned int data_size = input_length * sizeof(char);
	unsigned int histo_size = HISTO_SIZE * sizeof(unsigned int);

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU for the output image
	cudaMalloc((void**)&d_data, data_size);
	cudaMalloc((void**)&d_histo, histo_size);
	cudaMemset(d_histo, 0, histo_size);
	checkCUDAError("CUDA malloc");

	// allocate host data
	h_data = (char*)malloc(data_size);
	h_histo = (unsigned int*)malloc(histo_size);
	generate_data(h_data, input_length);

	// copy image to device memory
	cudaMemcpy(d_data, h_data, input_length, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");
	
	cudaEventRecord(start, 0);
	// Task 1.3 Add kernel launch here...

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel normal");


	// copy the histogram back from the GPU
	cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Execution time:");
	printf("\t%f\n", ms);

	// Task 1.4 Verify output using a CPU function
	

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_data);
	cudaFree(d_histo);
	free(h_data);
	free(h_histo);

	return 0;
}



