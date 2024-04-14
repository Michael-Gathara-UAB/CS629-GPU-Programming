#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void generate_data(float * data, unsigned int length) {
	for (int i = 0; i < length; ++i) {
		data[i] = (float)rand() / RAND_MAX;
	}
}

__global__ void GPU_scan(float * X, float * Y, unsigned int length) {
	// Task 2.1 Add your implementation here
}

void CPU_scan(float * X, float * Y, unsigned int length) {
	// Task 2.2 Add a CPU implementation for verification

}

/* Host code */
int main(void) {
	unsigned int input_length = 2048;
	float * h_input, * d_input, * h_output, * d_output;
	cudaEvent_t start, stop;
	float ms;

	unsigned int data_size = input_length * sizeof(float);

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU for the output image
	cudaMalloc((void**)&d_input, data_size);
	cudaMalloc((void**)&d_output, data_size);
	checkCUDAError("CUDA malloc");

	// allocate host data
	h_input = (float*)malloc(data_size);
	h_output = (float*)malloc(data_size);
	generate_data(h_input, input_length);

	// copy image to device memory
	cudaMemcpy(d_input, h_input, input_length, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");
	
	cudaEventRecord(start, 0);
	// Task 2.3 Add kernel launch here...

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel normal");


	// copy the histogram back from the GPU
	cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Execution time:");
	printf("\t%f\n", ms);

	// Task 2.4 Verify output using a CPU function
	

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);

	return 0;
}



