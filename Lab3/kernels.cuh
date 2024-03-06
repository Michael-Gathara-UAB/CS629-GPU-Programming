#ifndef KERNEL_H //ensures header is only included once
#define KERNEL_H

#define NUM_RECORDS 2048
#define THREADS_PER_BLOCK 256
#define FLT_MAX 3.402823466e+38F


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


__device__ volatile float d_max_mark = 0;
__device__ volatile int d_max_mark_student_id = 0;
__device__ int lock = 0;

// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_RECORDS) {
        float mark = d_records->assignment_marks[idx];
        int id = d_records->student_ids[idx];
        bool need_lock = true;

        while (need_lock) {
            if (atomicCAS(&lock, 0, 1) == 0) {
                if (mark > d_max_mark) {
                    d_max_mark = mark;
                    d_max_mark_student_id = id;
                }
                atomicExch(&lock, 0);
                need_lock = false;
            }
        }
    }
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
    extern __shared__ student_record s_records[];

	if (idx < NUM_RECORDS) {
		// Each thread loads one student record into shared memory
		s_records[threadIdx.x].student_id = d_records->student_ids[idx];
		s_records[threadIdx.x].assignment_mark = d_records->assignment_marks[idx];
	}
	__syncthreads();

	//Task 3.2) Reduce in shared memory in parallel
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
		int index = 2 * stride * threadIdx.x;
		if (index < blockDim.x) {
			if (s_records[index].assignment_mark < s_records[index + stride].assignment_mark) {
				s_records[index] = s_records[index + stride];
			}
		}
		__syncthreads(); 
	}

    if (threadIdx.x == 0) {
		d_reduced_records[blockIdx.x] = s_records[0];
	}
}

//Task 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Task 4.1) Complete the kernel
	int i = 2 * threadIdx.x;
	
	
}

#endif //KERNEL_H
