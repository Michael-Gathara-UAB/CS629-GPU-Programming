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


// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_RECORDS) { // Ensure we do not go out of bounds
        float mark = d_records->assignment_marks[idx];
        int id = d_records->student_ids[idx];
        bool need_lock = true;

        // Try to acquire lock to update global maximum
        while (need_lock) {
            if (atomicCAS(&lock, 0, 1) == 0) {
                // Lock acquired, now update the global maximum
                if (mark > d_max_mark) {
                    d_max_mark = mark;
                    d_max_mark_student_id = id;
                }
                // Release lock
                atomicExch(&lock, 0);
                need_lock = false; // exit the while loop
            }
            // Implicit else: wait and try to acquire the lock again
        }
    }
}



// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
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
	int i = 2 * threadIdx.x;
	
	
}

#endif //KERNEL_H
