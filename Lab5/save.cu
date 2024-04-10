#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IMAGE_DIM 2048

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

using uchar = unsigned char;

void output_image_file(uchar *image);
void input_image_file(char *filename, uchar3 *image);
void checkCUDAError(const char *msg);

__global__ void image_to_grayscale_naive(uchar3 *image, uchar *image_output) {
    // Add your implementation here
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = x + y * IMAGE_DIM;

    uchar3 pixel = image[i];

    // grayscale_output = r * 0.21 + g * 0.72 + b * 0.07
    image_output[i] = 0.21 * pixel.x + 0.72 * pixel.y + 0.07 * pixel.z;
}

__global__ void image_to_grayscale(uchar *r, uchar *g, uchar *b,
                                   uchar *image_output) {
    // Add your implementation here
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = x + y * IMAGE_DIM;

    // grayscale_output = r * 0.21 + g * 0.72 + b * 0.07
    image_output[i] =
        static_cast<uchar>(0.21 * r[i] + 0.72 * g[i] + 0.07 * b[i]);
}

/* Host code */

int main(void) {
    unsigned int image_size, image_output_size;
    uchar3 *d_image, *h_image;
    uchar *d_image_output, *h_image_output;
    cudaEvent_t start, stop;
    float ms;

    image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar3);
    image_output_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar);

    // create timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the GPU for the output image
    cudaMalloc((void **)&d_image, image_size);
    cudaMalloc((void **)&d_image_output, image_output_size);
    checkCUDAError("CUDA malloc");

    // allocate and load host image
    h_image = (uchar3 *)malloc(image_size);
    h_image_output = (uchar *)malloc(image_output_size);
    input_image_file("input.ppm", h_image);

    // copy image to device memory
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy to device");

    // cuda layout and execution
    dim3 blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
    dim3 threadsPerBlock(16, 16);

    // normal version
    cudaEventRecord(start, 0);
    image_to_grayscale_naive<<<blocksPerGrid, threadsPerBlock>>>(
        d_image, d_image_output);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    checkCUDAError("kernel normal");

    // copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_output_size,
               cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device");

    // output timings
    printf("Naive Execution time:");
    printf("\t%f\n", ms);

    // output image
    output_image_file(h_image_output);

    // cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_image_output);
    free(h_image);
    free(h_image_output);

    // the lab version
    cudaEvent_t start_lab, stop_lab;
    float ms_lab;
    uchar *d_r, *d_g, *d_b;
    uchar *h_r, *h_g, *h_b;

	h_r = (uchar *)malloc(IMAGE_DIM * IMAGE_DIM);
    h_g = (uchar *)malloc(IMAGE_DIM * IMAGE_DIM);
    h_b = (uchar *)malloc(IMAGE_DIM * IMAGE_DIM);
	uchar* lab_image_output = (uchar *)malloc(image_output_size);

	input_image_file_lab("input.ppm", h_image, h_r, h_g, h_b);

    cudaEventCreate(&start_lab);
    cudaEventCreate(&stop_lab);

    cudaMalloc((void **)&d_r, IMAGE_DIM * IMAGE_DIM);
    cudaMalloc((void **)&d_g, IMAGE_DIM * IMAGE_DIM);
    cudaMalloc((void **)&d_b, IMAGE_DIM * IMAGE_DIM);
    checkCUDAError("CUDA malloc for color channels");

    cudaMalloc((void **)&d_r, IMAGE_DIM * IMAGE_DIM);
    cudaMalloc((void **)&d_g, IMAGE_DIM * IMAGE_DIM);
    cudaMalloc((void **)&d_b, IMAGE_DIM * IMAGE_DIM);
    checkCUDAError("CUDA malloc for color channels");

    cudaMemcpy(d_r, h_r, IMAGE_DIM * IMAGE_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, IMAGE_DIM * IMAGE_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, IMAGE_DIM * IMAGE_DIM, cudaMemcpyHostToDevice);
    checkCUDAError("CUDA memcpy to device for color channels");

    cudaEventRecord(start_lab, 0);
    image_to_grayscale<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_g, d_b,
                                                           d_image_output);
    cudaEventRecord(stop_lab, 0);
    cudaEventSynchronize(stop_lab);
    cudaEventElapsedTime(&ms_lab, start_lab, stop_lab);
    checkCUDAError("kernel lab version");

    cudaMemcpy(h_image_output, d_image_output, image_output_size,
               cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy from device");

    printf("Lab Execution time:");
    printf("\t%f\n", ms_lab);

    output_image_file(h_image_output);

    return 0;
}

void output_image_file(uchar *image) {
    FILE *f;  // output file handle

    // open the output file and write header info for PPM filetype
    const char *input_file = "output.ppm";
    f = fopen(input_file, "wb");
    if (f == NULL) {
        fprintf(stderr, "Error opening 'output.ppm' output file\n");
        exit(1);
    }
    fprintf(f, "P5\n");
    fprintf(f, "# CS 629/729 Lab 05 Task01\n");
    fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
    for (int x = 0; x < IMAGE_DIM; x++) {
        for (int y = 0; y < IMAGE_DIM; y++) {
            int i = x + y * IMAGE_DIM;
            fwrite(&image[i], sizeof(unsigned char), 1,
                   f);  // only write garyscale
        }
    }

    fclose(f);
}

void input_image_file(char *filename, uchar3 *image) {
    FILE *f;  // input file handle
    char temp[256];
    unsigned int x, y, s;

    // open the input file and write header info for PPM filetype
    f = fopen("input.ppm", "rb");
    if (f == NULL) {
        fprintf(stderr, "Error opening 'input.ppm' input file\n");
        exit(1);
    }
    fscanf(f, "%s\n", temp);
    fscanf(f, "%d %d\n", &x, &y);
    fscanf(f, "%d\n", &s);
    if ((x != y) && (x != IMAGE_DIM)) {
        fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
        exit(1);
    }

    for (int x = 0; x < IMAGE_DIM; x++) {
        for (int y = 0; y < IMAGE_DIM; y++) {
            int i = x + y * IMAGE_DIM;
            fread(&image[i], sizeof(unsigned char), 3, f);
        }
    }

    fclose(f);
}

void input_image_file_lab(char *filename, uchar3 *image, uchar *r, uchar *g,
                          uchar *b) {
    FILE *f;  // input file handle
    char temp[256];
    unsigned int x, y, s;

    // open the input file and write header info for PPM filetype
    f = fopen("input.ppm", "rb");
    if (f == NULL) {
        fprintf(stderr, "Error opening 'input.ppm' input file\n");
        exit(1);
    }
    fscanf(f, "%s\n", temp);
    fscanf(f, "%d %d\n", &x, &y);
    fscanf(f, "%d\n", &s);
    if ((x != y) && (x != IMAGE_DIM)) {
        fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
        exit(1);
    }

    for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++) {
        r[i] = image[i].x;
        g[i] = image[i].y;
        b[i] = image[i].z;
    }

    fclose(f);
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
