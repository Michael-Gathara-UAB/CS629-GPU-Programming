#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HISTO_SIZE 256

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void generate_data(char* data, unsigned int length) {
    for (int i = 0; i < length; ++i) {
        data[i] = (char)rand() % HISTO_SIZE;
    }
}

// took this from the slides, week 11 slide 53
__global__ void GPU_histogram(char* data, unsigned int length,
                              unsigned int* histo) {
    // Task 1.1 Add your implementation here
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alph = data[i] - 'a';
        if (alph >= 0 && alph < 26) {
            atomicAdd(&histo[blockIdx.x * HISTO_SIZE + alph / 4], 1);
        }
        // if (alph >= 0 && alph < 26) {
        //     atomicAdd(&histo[alph / 4], 1);
        // }
    
    }
    if (blockIdx.x > 0) {
        __syncthreads();
        for (unsigned int bin = threadIdx.x; bin < HISTO_SIZE; bin += blockDim.x) {
            unsigned binVal = histo[blockIdx.x * HISTO_SIZE + bin];
            if (binVal > 0) {
                atomicAdd(&histo[bin], binVal);
            }
        }
    }
}

// took this from the slides, week 11 slide 38
void CPU_histogram(char* data, unsigned int length, unsigned int* histo) {
    // Task 1.2 Add a CPU implementation for verification
    for (unsigned int i = 0; i < length; ++i) {
        int alph = data[i] - 'a';
        if (alph >= 0 && alph < 26) {
            histo[alph / 4]++;
        } 

    }
}

/* Host code */
int main(void) {
    unsigned int input_length = 2048;
    char *h_data, *d_data;
    char* c_data;
    unsigned int* c_histo;
    unsigned int *h_histo, *d_histo;
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (input_length + threadsPerBlock - 1) / threadsPerBlock;
    // GPU_histogram << <threadsPerBlock, (input_length + threadsPerBlock - 1) /
    // threadsPerBlock>> > (d_data, input_length, d_histo);
    GPU_histogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, input_length,
                                                      d_histo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("kernel normal");

    // copy the histogram back from the GPU
    cudaMemcpy(h_histo, d_histo, histo_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device");

    // Task 1.4 Verify output using a CPU function
    c_data = (char*)malloc(data_size);
    c_histo = (unsigned int*)malloc(histo_size);
    memcpy(c_data, h_data, data_size);
    // memcpy(c_histo, h_histo, histo_size);
    CPU_histogram(c_data, input_length, c_histo);

    for (int i = 0; i < HISTO_SIZE; ++i) {
        if (h_histo[i] != c_histo[i]) {
            printf("Error: GPU x CPU mismatch @ %d which is %d and %d\n", i,
                   h_histo[i], c_histo[i]);
            break;
        }
    }

	// THESE ARE FOR MY DEBUGGING
    // for (auto i = 0; i < input_length; ++i) {
    // 	printf("%d: %d\n", i, h_data[i]);
    // }

    // for (int i = 0; i < HISTO_SIZE; ++i) {
    //     printf("%d: %d\n", i, h_histo[i]);
    // }
    // printf("----------CPU----------\n");
    // for (int i = 0; i < HISTO_SIZE; ++i) {
    //     printf("%d: %d\n", i, c_histo[i]);
    // }

    // output timings
    printf("Execution time:");
    printf("\t%f\n", ms);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_histo);
    free(h_data);
    free(h_histo);
    free(c_data);
    free(c_histo);

    return 0;
}