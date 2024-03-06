#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// include kernels and cuda headers after definitions of structures
#include "kernels.cuh" 


void checkCUDAError(const char*);
void readRecords(student_record *records);
void studentRecordAOS2SOA(student_record *aos, student_records *soa);
void maximumMark_atomic(student_records*, student_records*, student_records*, student_records*);
void maximumMark_recursive(student_records*, student_records*, student_records*, student_records*);
void maximumMark_SM(student_records*, student_records*, student_records*, student_records*);
void maximumMark_shuffle(student_records*, student_records*, student_records*, student_records*);


int main(void) {
	student_record *recordsAOS;
	student_records *h_records;
	student_records *h_records_result;
	student_records *d_records;
	student_records *d_records_result;
	
	//host allocation
	recordsAOS = (student_record*)malloc(sizeof(student_record)*NUM_RECORDS);
	h_records = (student_records*)malloc(sizeof(student_records));
	h_records_result = (student_records*)malloc(sizeof(student_records));

	//device allocation
	cudaMalloc((void**)&d_records, sizeof(student_records));
	cudaMalloc((void**)&d_records_result, sizeof(student_records));
	checkCUDAError("CUDA malloc");

	//read file
	readRecords(recordsAOS);
	studentRecordAOS2SOA(recordsAOS, h_records);
	
	//free AOS as it is no longer needed
	free(recordsAOS);

	//apply each approach in turn 
	maximumMark_atomic(h_records, h_records_result, d_records, d_records_result);
	// maximumMark_recursive(h_records, h_records_result, d_records, d_records_result);
	// maximumMark_SM(h_records, h_records_result, d_records, d_records_result);
	// maximumMark_shuffle(h_records, h_records_result, d_records, d_records_result);


	// Cleanup
	free(h_records);
	free(h_records_result);
	cudaFree(d_records);
	cudaFree(d_records_result);
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

void readRecords(student_record *records){
	FILE *f = NULL;
	f = fopen("Student.dat", "rb"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find file \n");
		exit(1);
	}

	//read student data
	if (fread(records, sizeof(student_record), NUM_RECORDS, f) != NUM_RECORDS){
		fprintf(stderr, "Error: Unexpected end of file!\n");
		exit(1);
	}
	fclose(f);
}

void studentRecordAOS2SOA(student_record *aos, student_records *soa){
	for (int i = 0; i < NUM_RECORDS; i++) {
		soa->student_ids[i] = aos[i].student_id;
		soa->assignment_marks[i] = aos[i].assignment_mark;
	}
}

void maximumMark_atomic(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Atomics: CUDA memcpy");

	cudaEventRecord(start, 0);

	// Task 1.2 Confgure the kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (NUM_RECORDS + threadsPerBlock - 1) / threadsPerBlock;

	// Task 1.3) Launch and synchronize the kernel
	cudaDeviceSynchronize();
	maximumMark_atomic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_records);
	cudaDeviceSynchronize();
	
	// Task 1.4)  Copy result back to host
	cudaMemcpyFromSymbol(&max_mark, d_max_mark, sizeof(float), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&max_mark_student_id, d_max_mark_student_id, sizeof(int), 0, cudaMemcpyDeviceToHost);

	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// Task 1.5) Use CPU to validate results

	float cpuMaxMark = 0.0f;
	int cpuMaxMarkStudentId = -1;
	for (int i = 0; i < NUM_RECORDS; i++) {
		if (h_records->assignment_marks[i] > cpuMaxMark) {
			cpuMaxMark = h_records->assignment_marks[i];
			cpuMaxMarkStudentId = h_records->student_ids[i];
		}
	}

	// printf("CPU: Highest mark recorded %f was by student %d\n", cpuMaxMark, cpuMaxMarkStudentId);

	// //output result
	// printf("Atomics: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	// printf("\tExecution time was %f ms\n", time);
	printf("%d", max_mark_student_id);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records, int num_records) {
    extern __shared__ student_record sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Task 2.1) Load student records into shared memory
    if (i < num_records) {
        sdata[tid].assignment_mark = d_records->assignment_marks[i];
        sdata[tid].student_id = d_records->student_ids[i];
        if (i + blockDim.x < num_records) {
            float nextMark = d_records->assignment_marks[i + blockDim.x];
            if (nextMark > sdata[tid].assignment_mark) {
                sdata[tid].assignment_mark = nextMark;
                sdata[tid].student_id = d_records->student_ids[i + blockDim.x];
            }
        }
    } else {
        sdata[tid].assignment_mark = -1.0;
        sdata[tid].student_id = -1;
    }
    __syncthreads();

    // Task 2.2) Compare two values and write the result to d_reduced_records
    // for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    //     if (tid < s && (i + s) < num_records) {
    //         if (sdata[tid + s].assignment_mark > sdata[tid].assignment_mark) {
    //             sdata[tid].assignment_mark = sdata[tid + s].assignment_mark;
    //             sdata[tid].student_id = sdata[tid + s].student_id;
    //         }
    //     }
    //     __syncthreads();
    // }

    // Write the result for this block to d_reduced_records
    if (tid == 0) {
        d_reduced_records->assignment_marks[blockIdx.x] = sdata[0].assignment_mark;
        d_reduced_records->student_ids[blockIdx.x] = sdata[0].student_id;
    }
}


//Task 3)
void maximumMark_SM(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("SM: CUDA memcpy");

	cudaEventRecord(start, 0);
	
	//Task 3.4) Call the shared memory reduction kernel
    //Task 3.5) Copy the final block values back to CPU
    cudaMemcpy(h_records_result->assignment_marks, d_reduced_records, blocksPerGrid * sizeof(student_record), cudaMemcpyDeviceToHost);

    //Task 3.6) Reduce the block level results on CPU
    for (i = 0; i < blocksPerGrid; i++) {
        if (h_records_result->assignment_marks[i] > max_mark) {
            max_mark = h_records_result->assignment_marks[i];
            max_mark_student_id = h_records_result->student_ids[i];
        }
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("SM: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Task 4)
void maximumMark_shuffle(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	unsigned int i;
	unsigned int warps_per_grid;
	float max_mark;
	int max_mark_student_id;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Shuffle: CUDA memcpy");
	
	cudaEventRecord(start, 0);

	//Task 4.2) Execute the kernel, copy back result, reduce final values on CPU

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("Shuffle: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
