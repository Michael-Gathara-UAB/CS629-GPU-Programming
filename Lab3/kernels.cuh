#ifndef KERNEL_H  // ensures header is only included once
#define KERNEL_H

#define NUM_RECORDS 1048576
#define THREADS_PER_BLOCK 256
#define FLT_MAX 3.402823466e+38F

struct student_record {
    int student_id;
    float assignment_mark;
};

struct student_records {
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

// Task 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(
    student_records *d_records, student_records *d_reduced_records) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ student_record shared_records[THREADS_PER_BLOCK];

    if (idx < NUM_RECORDS) {
        shared_records[threadIdx.x].student_id = d_records->student_ids[idx];
        shared_records[threadIdx.x].assignment_mark =
            d_records->assignment_marks[idx];
    } else {
        shared_records[threadIdx.x].student_id = -1;
        shared_records[threadIdx.x].assignment_mark = -1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && idx + s < NUM_RECORDS) {
            if (shared_records[threadIdx.x].assignment_mark <
                shared_records[threadIdx.x + s].assignment_mark) {
                shared_records[threadIdx.x] = shared_records[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_reduced_records->student_ids[blockIdx.x] =
            shared_records[0].student_id;
        d_reduced_records->assignment_marks[blockIdx.x] =
            shared_records[0].assignment_mark;
    }
}

// Task 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records,
                                      student_records *d_reduced_records) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ student_record shared_records[THREADS_PER_BLOCK];

    // 3.1) Load a single student record into shared memory
    if (idx < NUM_RECORDS) {
        shared_records[threadIdx.x].student_id = d_records->student_ids[idx];
        shared_records[threadIdx.x].assignment_mark =
            d_records->assignment_marks[idx];
    } else {
        shared_records[threadIdx.x].student_id = -1;
        shared_records[threadIdx.x].assignment_mark = -1;
    }
    __syncthreads();  

    // 3.2) Reduce in shared memory in parallel
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_records[threadIdx.x].assignment_mark <
                shared_records[threadIdx.x + s].assignment_mark) {
                shared_records[threadIdx.x] = shared_records[threadIdx.x + s];
            }
        }
        __syncthreads(); 
    }

    // 3.3) Write the result for the first thread in the block
    if (threadIdx.x == 0) {
        d_reduced_records->student_ids[blockIdx.x] =
            shared_records[0].student_id; 
        d_reduced_records->assignment_marks[blockIdx.x] =
            shared_records[0].assignment_mark; 
    }
}

// Task 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize; 
    int lane_id = threadIdx.x % warpSize;

    float mark = (idx < NUM_RECORDS) ? d_records->assignment_marks[idx] : -1;
    int id = (idx < NUM_RECORDS) ? d_records->student_ids[idx] : -1;


    // The slides did it without a loop but that seems like too long since only the "offset" changes
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float mark_next = __shfl_down_sync(0xFFFFFFFF, mark, offset);
        int id_next = __shfl_down_sync(0xFFFFFFFF, id, offset);

        if (mark_next > mark) {
            mark = mark_next;
            id = id_next;
        }
    }

    if (lane_id == 0) {
        d_reduced_records->assignment_marks[blockIdx.x * (blockDim.x / warpSize) + warpId] = mark;
        d_reduced_records->student_ids[blockIdx.x * (blockDim.x / warpSize) + warpId] = id;
    }
}


#endif  // KERNEL_H
