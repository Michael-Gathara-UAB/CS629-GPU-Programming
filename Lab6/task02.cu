#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void generate_data(float *data, unsigned int length) {
    for (int i = 0; i < length; ++i) {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// https://www.youtube.com/watch?v=1G4jfLcnI2w
// https://vccvisualization.org/teaching/CS380/CS380_fall2021_lecture_23.pdf
// https://homepages.dcc.ufmg.br/~sylvain.collange/gpucourse/gpu_ufmg_2015_0.pdf
// This is the brent-kung version
// __global__ void GPU_scan(float *X, float *Y, unsigned int length) {
//     // Task 2.1 Add your implementation here
//     extern __shared__ float temp[];
//     int offset = 1;

//     // int first_check = threadIdx.x;
//     // int second_check = threadIdx.x + (length / 2);
//     // if (first_check < length) {
//     //     temp[first_check] = X[first_check];
//     // } else {
//     //     temp[first_check] = 0;
//     // }

//     // if (second_check < length) {
//     //     temp[second_check] = X[second_check];
//     // } else {
//     //     temp[second_check] = 0;
//     // }

// 	int idx = threadIdx.x;
//     int blockOffset = blockIdx.x * blockDim.x;

//     // Load elements into shared memory
//     int first_check = blockOffset + idx;
//     int second_check = blockOffset + idx + blockDim.x;

//     if (first_check < length) {
//         temp[idx] = X[first_check];
//     } else {
//         temp[idx] = 0;
//     }

//     if (second_check < length) {
//         temp[idx + blockDim.x] = X[second_check];
//     } else {
//         temp[idx + blockDim.x] = 0;
//     }

//     // int stride = length / 2;
//     // while (stride > 0) {
//     //     __syncthreads();
//     //     int index =
//     //         threadIdx.x * stride *
//     //         2;
//     //     if (index + stride < length) {
//     //         int ai = index + stride - 1;
//     //         int bi = index + (stride * 2) - 1;

//     //         temp[bi] += temp[ai];
//     //     }
//     //     stride /= 2;
//     // }

//     // for (int d = length >> 1; d > 0; d >>= 1) {
//     //     __syncthreads();
//     //     if (threadIdx.x < d) {
//     //         int first_check = offset * (2 * threadIdx.x + 1) - 1;
//     //         int second_check = offset * (2 * threadIdx.x + 2) - 1;

//     //         temp[second_check] += temp[first_check];
//     //     }
//     //     offset *= 2;
//     // }

// 	for (int d = length >> 1; d > 0; d >>= 1) {
//         __syncthreads();
//         if (idx < d) {
//             int ai = offset * (2 * idx + 1) - 1;
//             int bi = offset * (2 * idx + 2) - 1;
//             if (bi < length) {
//                 temp[bi] += temp[ai];
//             }
//         }
//         offset *= 2;
//     }

//     if (threadIdx.x == 0) {
//         temp[length - 1] = 0;
//     }

//     for (int d = 1; d < length; d *= 2) {
//         offset >>= 1;
//         __syncthreads();
//         if (threadIdx.x < d) {
//             int first_check = offset * (2 * threadIdx.x + 1) - 1;
//             int second_check = offset * (2 * threadIdx.x + 2) - 1;

//             float t = temp[first_check];
//             temp[first_check] = temp[second_check];
//             temp[second_check] += t;
//         }
//     }
//     __syncthreads();

//     // if (first_check < length) {
//     //     Y[first_check] = temp[first_check];
//     // }

//     // if (second_check < length) {
//     //     Y[second_check] = temp[second_check];
//     // }
// 	if (first_check < length) {
//         Y[first_check] = temp[idx];
//     }
//     if (second_check < length) {
//         Y[second_check] = temp[idx + blockDim.x];
//     }
// }

__global__ void GPU_scan(float *X, float *Y, unsigned int length) {
    extern __shared__ float temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory.
    // This is assuming one element per thread
    int ai = thid;
    int bi = thid + (length / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = X[ai];
    temp[bi + bankOffsetB] = X[bi];

    for (int d = length >> 1; d > 0;
         d >>= 1) {  // build sum in place up the tree
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) {
        temp[length - 1 + CONFLICT_FREE_OFFSET(length - 1)] = 0;
    }  // clear the last element

    for (int d = 1; d < length; d *= 2) {  // traverse down tree & build scan
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // write results to device memory
    Y[ai] = temp[ai + bankOffsetA];
    Y[bi] = temp[bi + bankOffsetB];
}

void CPU_scan(float *X, float *Y, unsigned int length) {
    // Task 2.2 Add a CPU implementation for verification
    Y[0] = X[0];
    for (int i = 1; i < length; i++) {
        Y[i] = Y[i - 1] + X[i];
    }
}

/* Host code */
int main(void) {
    unsigned int input_length = 2048;
    float *h_input, *d_input, *h_output, *d_output;
    cudaEvent_t start, stop;
    float ms;

    unsigned int data_size = input_length * sizeof(float);

    // create timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the GPU for the output image
    cudaMalloc((void **)&d_input, data_size);
    cudaMalloc((void **)&d_output, data_size);
    checkCUDAError("CUDA malloc");

    // allocate host data
    h_input = (float *)malloc(data_size);
    h_output = (float *)malloc(data_size);
    generate_data(h_input, input_length);

    // copy image to device memory
    cudaMemcpy(d_input, h_input, input_length, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy to device");

    cudaEventRecord(start, 0);
    // Task 2.3 Add kernel launch here...
    // dim3 block(256, 1);
    // dim3 grid((input_length + block.x - 1) / block.x, 1);
    // size_t sharedMemSize =
    //     block.x * sizeof(float);
    // GPU_scan<<<grid, block, sharedMemSize>>>(d_input, d_output,
    // input_length);
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (input_length + threadsPerBlock - 1) /
    // threadsPerBlock; size_t sharedMemSize = threadsPerBlock * sizeof(float);
    // GPU_scan<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input,
    // d_output, input_length);

    // Launch the kernel
    dim3 block(256, 1);  // or whatever block size makes sense for your hardware
    dim3 grid((input_length + block.x - 1) / block.x, 1);
    size_t sharedMemSize =
        block.x * sizeof(float) * 2;  // 2 * is for double buffering
    GPU_scan<<<grid, block, sharedMemSize>>>(d_input, d_output, input_length);

    // GPU_scan<<<blocksPerGrid, threadsPerBlock, data_size >>> (d_input,
    // d_output, input_length); int threadsPerBlock = 256; int blocksPerGrid =
    // (input_length + threadsPerBlock - 1) / threadsPerBlock; size_t
    // sharedMemSize = threadsPerBlock * sizeof(float);
    // GPU_scan<<<blocksPerGrid, threadsPerBlock, data_size>>>(
    //     d_input, d_output, input_length);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("kernel normal");

    // copy the histogram back from the GPU
    cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device");

    // Task 2.4 Verify output using a CPU function
    // float *c_data = (float *)malloc(data_size);
    // memcpy(c_data, h_input, data_size);
    // CPU_scan(c_data, c_data, input_length);
    // CPU_scan(h_input, c_data, input_length);

    float *c_output = (float *)malloc(data_size);
    CPU_scan(h_input, c_output, input_length);

    for (unsigned int i = 0; i < input_length; ++i) {
        if (fabs(c_output[i] - h_output[i]) > 1e-5) {
            printf("Error: GPU x CPU mismatch @ %d which is %f and %f\n", i,
                   h_output[i], c_output[i]);
            break;
        }
    }

    // output timings
    printf("Execution time:");
    printf("\t%f\n", ms);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
