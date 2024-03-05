#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

#define NUM_RECORDS 2048
#define THREADS_PER_BLOCK 256


struct student_record{
	int student_id;
	float assignment_mark;
};

struct student_records{
	int student_ids[NUM_RECORDS];
	float assignment_marks[NUM_RECORDS];
};

typedef struct student_record student_record;
typedef struct student_records student_records;

__device__ float d_max_mark = 0;
__device__ int d_max_mark_student_id = 0;


// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float mark = d_records->assignment_marks[idx];
	int id = d_records->student_ids[idx];

	// Task 1.1) Use atomicCAS function to create a critical section that updates d_max_mark and d_max_mark_student_id

}

//Task 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Task 2.1) Load a single student record into shared memory

	//Task 2.2) Compare two values and write the result to d_reduced_records

}


//Task 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Task 3.1) Load a single student record into shared memory

	//Task 3.2) Reduce in shared memory in parallel

	//Task 3.3) Write the result

}

//Task 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Task 4.1) Complete the kernel
	
}

#endif //KERNEL_H