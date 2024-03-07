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
void checkMaxCPU(student_records*);

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS (NUM_RECORDS+THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK

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
	maximumMark_recursive(h_records, h_records_result, d_records, d_records_result);
	maximumMark_SM(h_records, h_records_result, d_records, d_records_result);
	maximumMark_shuffle(h_records, h_records_result, d_records, d_records_result);


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
	// Task 0.1 Convert student_record to student_records
    for(int i = 0; i<NUM_RECORDS; i++){
        soa->student_ids[i] = aos[i].student_id;
        soa->assignment_marks[i] = aos[i].assignment_mark;
    }
}

void checkMaxCPU(student_records *h_records){
    float max_cpu = 0;
    int max_cpu_id = 0;
    for (int i =0;i<NUM_RECORDS;i++){
        if(h_records->assignment_marks[i]>=max_cpu){
            max_cpu = h_records->assignment_marks[i];
            max_cpu_id = h_records->student_ids[i];
        }
    }
    printf("CPU : Highest mark recorded %f was by student %d\n", max_cpu, max_cpu_id);
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
    
   // int threadsPerBlock = 256;
   // int numBlocks = (NUM_RECORDS + threadsPerBlock -1)/threadsPerBlock;
   // defined outside 
   
	// Task 1.3) Launch and synchronize the kernel
    
    maximumMark_atomic_kernel<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_records);
    
	cudaDeviceSynchronize();
    
	// Task 1.4)  Copy result back to host
	
    cudaMemcpyFromSymbol(&max_mark,d_max_mark,sizeof(float),0,cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&max_mark_student_id,d_max_mark_student_id,sizeof(float),0,cudaMemcpyDeviceToHost);
    
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// Task 1.5) Use CPU to validate results
    checkMaxCPU(h_records);
    
	//output result
	printf("Atomics: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

//Task 2)
void maximumMark_recursive(student_records *h_records, student_records *h_records_result, student_records *d_records, student_records *d_records_result){
	int i;
	float max_mark;
	int max_mark_student_id;
	student_records *d_records_temp;
	float time;
	cudaEvent_t start, stop;
	
	max_mark = 0;
	max_mark_student_id = 0.0f;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//memory copy records to device
	cudaMemcpy(d_records, h_records, sizeof(student_records), cudaMemcpyHostToDevice);
	checkCUDAError("Recursive: CUDA memcpy");

	cudaEventRecord(start, 0);
	
    
	//Task 2.3) Recursively call GPU steps until there are THREADS_PER_BLOCK values left
    
   int records = NUM_RECORDS;
    
    while (records>=THREADS_PER_BLOCK){
        int num_blocks = (records + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK;
        maximumMark_recursive_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_records, d_records_result);
        cudaDeviceSynchronize();
        d_records_temp = d_records;
        d_records = d_records_result;
        d_records_result = d_records_temp;
        records = records/2;
    }
    
    //Task 2.4) copy back the final THREADS_PER_BLOCK values
    cudaMemcpy(h_records_result, d_records, sizeof(student_records), cudaMemcpyDeviceToHost);
    
	//Task 2.5) reduce the final THREADS_PER_BLOCK values on CPU
    for (i = 0; i< records; i++){
        if(h_records_result->assignment_marks[i] >= max_mark){
            max_mark = h_records_result->assignment_marks[i];
            max_mark_student_id = h_records_result->student_ids[i];
        }
    }

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output the result
	printf("Recursive: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
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
    
    maximumMark_SM_kernel<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_records, d_records_result);
    cudaDeviceSynchronize();
    
	//Task 3.5) Copy the final block values back to CPU
    
    cudaMemcpy(h_records_result, d_records_result,  sizeof(student_records), cudaMemcpyDeviceToHost);
    
	//Task 3.6) Reduce the block level results on CPU
    
    for (i = 0; i< NUM_RECORDS/2; i++){
        if(h_records_result->assignment_marks[i] >= max_mark){
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
    
    warps_per_grid = NUM_BLOCKS;
    maximumMark_shuffle_kernel<<<warps_per_grid,THREADS_PER_BLOCK>>>(d_records, d_records_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_records_result,d_records_result,sizeof(student_records), cudaMemcpyDeviceToHost);
    
    for (i = 0; i< NUM_RECORDS; i++){
        if(h_records_result->assignment_marks[i] >= max_mark){
            max_mark = h_records_result->assignment_marks[i];
            max_mark_student_id = h_records_result->student_ids[i];
        }
    }
    
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	//output result
	printf("Shuffle: Highest mark recorded %f was by student %d\n", max_mark, max_mark_student_id);
	printf("\tExecution time was %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
