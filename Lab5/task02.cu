#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>


#define IMAGE_DIM 2048
constexpr int radius_one = 1;
constexpr int radius_two = 2;
constexpr int radius_three = 3;

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

using uchar = unsigned char; // from task1

void output_image_file(uchar3* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);


__global__ void image_blur_A(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	extern __shared__ uchar3 shared_image[];

    int local_x = threadIdx.x + radius_one;
    int local_y = threadIdx.y + radius_one;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = global_x + global_y * IMAGE_DIM;
    int local_idx = local_x + local_y * (blockDim.x + 2 * radius_one);

    shared_image[local_idx] = image[global_idx];

    if (threadIdx.x < radius_one) {
        int halo_idx = (global_x - radius_one + IMAGE_DIM) % IMAGE_DIM + global_y * IMAGE_DIM;
        shared_image[local_idx - radius_one] = image[halo_idx];
    }
    
	if (threadIdx.x >= blockDim.x - radius_one) {
        int halo_idx = (global_x + radius_one) % IMAGE_DIM + global_y * IMAGE_DIM;
        shared_image[local_idx + radius_one] = image[halo_idx];
    }
    
	if (threadIdx.y < radius_one) {
        int halo_idx = global_x + ((global_y - radius_one + IMAGE_DIM) % IMAGE_DIM) * IMAGE_DIM;
        shared_image[local_idx - radius_one * (blockDim.x + 2 * radius_one)] = image[halo_idx];
    }

    if (threadIdx.y >= blockDim.y - radius_one) {
        int halo_idx = global_x + ((global_y + radius_one) % IMAGE_DIM) * IMAGE_DIM;
        shared_image[local_idx + radius_one * (blockDim.x + 2 * radius_one)] = image[halo_idx];
    }

    __syncthreads();

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float factor = 1.0f / ((radius_one * 2 + 1) * (radius_one * 2 + 1));

    for (int dy = -radius_one; dy <= radius_one; dy++) {
        for (int dx = -radius_one; dx <= radius_one; dx++) {
            int idx = (local_x + dx) + (local_y + dy) * (blockDim.x + 2 * radius_one);
            uchar3 pixel = shared_image[idx];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }
    }

    sum.x *= factor;
    sum.y *= factor;
    sum.z *= factor;

    if (local_x < blockDim.x && local_y < blockDim.y) {  
        uchar3 output_pixel;
        output_pixel.x = static_cast<uchar>(sum.x);
        output_pixel.y = static_cast<uchar>(sum.y);
        output_pixel.z = static_cast<uchar>(sum.z);
        image_output[global_idx] = output_pixel;
    }
}

__global__ void image_blur_B(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	extern __shared__ uchar3 shared_image[];

    int local_x = threadIdx.x + radius_two;
    int local_y = threadIdx.y + radius_two;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = global_x + global_y * IMAGE_DIM;
    int local_idx = local_x + local_y * (blockDim.x + 2 * radius_two);

    shared_image[local_idx] = image[global_idx];

    if (threadIdx.x < radius_two) {
        int halo_idx = (global_x - radius_two + IMAGE_DIM) % IMAGE_DIM + global_y * IMAGE_DIM;
        shared_image[local_idx - radius_two] = image[halo_idx];
    }
    
	if (threadIdx.x >= blockDim.x - radius_two) {
        int halo_idx = (global_x + radius_two) % IMAGE_DIM + global_y * IMAGE_DIM;
        shared_image[local_idx + radius_two] = image[halo_idx];
    }
    
	if (threadIdx.y < radius_two) {
        int halo_idx = global_x + ((global_y - radius_two + IMAGE_DIM) % IMAGE_DIM) * IMAGE_DIM;
        shared_image[local_idx - radius_two * (blockDim.x + 2 * radius_two)] = image[halo_idx];
    }

    if (threadIdx.y >= blockDim.y - radius_two) {
        int halo_idx = global_x + ((global_y + radius_two) % IMAGE_DIM) * IMAGE_DIM;
        shared_image[local_idx + radius_two * (blockDim.x + 2 * radius_two)] = image[halo_idx];
    }

    __syncthreads();

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float factor = 1.0f / ((radius_two * 2 + 1) * (radius_two * 2 + 1));

    for (int dy = -radius_two; dy <= radius_two; dy++) {
        for (int dx = -radius_two; dx <= radius_two; dx++) {
            int idx = (local_x + dx) + (local_y + dy) * (blockDim.x + 2 * radius_two);
            uchar3 pixel = shared_image[idx];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }
    }

    sum.x *= factor;
    sum.y *= factor;
    sum.z *= factor;

    if (local_x < blockDim.x && local_y < blockDim.y) {  
        uchar3 output_pixel;
        output_pixel.x = static_cast<uchar>(sum.x);
        output_pixel.y = static_cast<uchar>(sum.y);
        output_pixel.z = static_cast<uchar>(sum.z);
        image_output[global_idx] = output_pixel;
    }

	
}

__global__ void image_blur_C(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	extern __shared__ uchar3 shared_image[];

    int local_x = threadIdx.x + radius_three;
    int local_y = threadIdx.y + radius_three;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = global_x + global_y * IMAGE_DIM;
    int local_idx = local_x + local_y * (blockDim.x + 2 * radius_three);

    shared_image[local_idx] = image[global_idx];

    if (threadIdx.x < radius_three) {
        int halo_idx = (global_x - radius_three + IMAGE_DIM) % IMAGE_DIM + global_y * IMAGE_DIM;
        shared_image[local_idx - radius_three] = image[halo_idx];
    }
    
	if (threadIdx.x >= blockDim.x - radius_three) {
        int halo_idx = (global_x + radius_three) % IMAGE_DIM + global_y * IMAGE_DIM;
        shared_image[local_idx + radius_three] = image[halo_idx];
    }
    
	if (threadIdx.y < radius_three) {
        int halo_idx = global_x + ((global_y - radius_three + IMAGE_DIM) % IMAGE_DIM) * IMAGE_DIM;
        shared_image[local_idx - radius_three * (blockDim.x + 2 * radius_three)] = image[halo_idx];
    }

    if (threadIdx.y >= blockDim.y - radius_three) {
        int halo_idx = global_x + ((global_y + radius_three) % IMAGE_DIM) * IMAGE_DIM;
        shared_image[local_idx + radius_three * (blockDim.x + 2 * radius_three)] = image[halo_idx];
    }

    __syncthreads();

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float factor = 1.0f / ((radius_three * 2 + 1) * (radius_three * 2 + 1));

    for (int dy = -radius_three; dy <= radius_three; dy++) {
        for (int dx = -radius_three; dx <= radius_three; dx++) {
            int idx = (local_x + dx) + (local_y + dy) * (blockDim.x + 2 * radius_three);
            uchar3 pixel = shared_image[idx];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }
    }

    sum.x *= factor;
    sum.y *= factor;
    sum.z *= factor;

    if (local_x < blockDim.x && local_y < blockDim.y) {  
        uchar3 output_pixel;
        output_pixel.x = static_cast<uchar>(sum.x);
        output_pixel.y = static_cast<uchar>(sum.y);
        output_pixel.z = static_cast<uchar>(sum.z);
        image_output[global_idx] = output_pixel;
    }
}


/* Host code */

int main(void) {
	unsigned int image_size;
	uchar3 *d_image, *d_image_output;
	uchar3 *h_image;
	cudaEvent_t start, stop;
	float ms;

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar3);

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU for the output image
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_image_output, image_size);
	checkCUDAError("CUDA malloc");

	// allocate and load host image
	h_image = (uchar3*)malloc(image_size);
	input_image_file("input.ppm", h_image);

	// copy image to device memory
	cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	//cuda layout and execution
	dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3    threadsPerBlock(16, 16);

	// normal version
	cudaEventRecord(start, 0);
	// Uncomment each line to test each kernel
	// image_blur_A << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output);
	image_blur_A<<<blocksPerGrid, threadsPerBlock, (16 + 2 * radius_one) * (16 + 2 * radius_one) * sizeof(uchar3)>>>(d_image, d_image_output);

	// image_blur_B << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output);
	image_blur_B<<<blocksPerGrid, threadsPerBlock, (16 + 2 * radius_two) * (16 + 2 * radius_two) * sizeof(uchar3)>>>(d_image, d_image_output);

	// image_blur_C << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output);
	image_blur_C<<<blocksPerGrid, threadsPerBlock, (16 + 2 * radius_three) * (16 + 2 * radius_three) * sizeof(uchar3)>>>(d_image, d_image_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	checkCUDAError("kernel normal");


	// copy the image back from the GPU for output to file
	cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Execution time:");
	printf("\t%f\n", ms);

	// output image
	output_image_file(h_image);

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_image_output);
	free(h_image);

	return 0;
}

void output_image_file(uchar3* image)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# CS 629/729 Lab 05 Task02\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);
}

void input_image_file(char* filename, uchar3* image)
{
	FILE *f; //input file handle
	char temp[256];
	unsigned int x, y, s;

	//open the input file and write header info for PPM filetype
	f = fopen("input.ppm", "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'input.ppm' input file\n");
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d %d\n", &x, &y);
	fscanf(f, "%d\n",&s);
	if ((x != y) && (x != IMAGE_DIM)){
		fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
		exit(1);
	}

	for (int x = 0; x < IMAGE_DIM; x++){
		for (int y = 0; y < IMAGE_DIM; y++){
			int i = x + y*IMAGE_DIM;
			fread(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);
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
