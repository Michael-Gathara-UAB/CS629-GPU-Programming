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

__device__ volatile float d_max_mark = 0;
__device__ volatile int d_max_mark_student_id = 0;

__device__ int lock = 0;

// Naive atomic implementation
__global__ void maximumMark_atomic_kernel(student_records *d_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float mark = d_records->assignment_marks[idx];
    int id = d_records->student_ids[idx];

    // Task 1.1) Use atomicCAS function to create a critical section that updates d_max_mark and d_max_mark_student_id
    
    bool need_lock = true;
    while (need_lock) {
        if (atomicCAS(&lock, 0, 1)==0) {
           if(mark>=d_max_mark){
                d_max_mark = mark;
                d_max_mark_student_id = id;
            }
            atomicExch(&lock, 0);
            need_lock = false;
        }
    }
}

//Task 2) Recursive Reduction
__global__ void maximumMark_recursive_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Task 2.1) Load a single student record into shared memory
    
    __shared__ student_record shared_sr[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    shared_sr[tid].student_id = d_records->student_ids[idx];
    shared_sr[tid].assignment_mark = d_records->assignment_marks[idx];
    __syncthreads();
    
	//Task 2.2) Compare two values and write the result to d_reduced_records
    
    if(tid%2 == 0 && tid+1<THREADS_PER_BLOCK){
        if(shared_sr[tid].assignment_mark < shared_sr[tid+1].assignment_mark){
            d_reduced_records->student_ids[idx/2] = shared_sr[tid+1].student_id;
            d_reduced_records->assignment_marks[idx/2] = shared_sr[tid+1].assignment_mark;
        }else{
            d_reduced_records->student_ids[idx/2] = shared_sr[tid].student_id;
            d_reduced_records->assignment_marks[idx/2] = shared_sr[tid].assignment_mark;
        }
    }
    
    __syncthreads();  
}


//Task 3) Using block level reduction
__global__ void maximumMark_SM_kernel(student_records *d_records, student_records *d_reduced_records) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Task 3.1) Load a single student record into shared memory
    
    __shared__ student_record shared_sr[NUM_RECORDS];
    
    int tid = threadIdx.x;
    shared_sr[tid].student_id = d_records->student_ids[idx];
    shared_sr[tid].assignment_mark = d_records->assignment_marks[idx];
    __syncthreads();
    
	//Task 3.2) Reduce in shared memory in parallel
    for (int stride = 1; stride<NUM_RECORDS;stride*=2){
        if(tid%stride == 0 && tid+stride<NUM_RECORDS){
            if(shared_sr[tid].assignment_mark < shared_sr[tid+stride].assignment_mark){
                shared_sr[tid] = shared_sr[tid+stride];
            }
        }
        __syncthreads();
    }
    
	//Task 3.3) Write the result
    if (tid==0){
        d_reduced_records->student_ids[idx] = shared_sr[0].student_id;
        d_reduced_records->assignment_marks[idx] = shared_sr[0].assignment_mark;
    }
    
}

//Task 4) Using warp level reduction
__global__ void maximumMark_shuffle_kernel(student_records *d_records, student_records *d_reduced_records) {
	//Task 4.1) Complete the kernel
    
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    
    float mark = d_records->assignment_marks[idx];
    int id = d_records->student_ids[idx];
    
    float mark_next = __shfl_down_sync(0xFFFFFFFF, mark, 16);
    float id_next = __shfl_down_sync(0xFFFFFFFF, id, 16);
    
    if (mark_next >= mark){
        mark = mark_next;
        id = id_next;
    }
    
    mark_next = __shfl_down_sync(0xFFFFFFFF, mark, 8);
    id_next = __shfl_down_sync(0xFFFFFFFF, id, 8);
    
    if (mark_next >= mark){
        mark = mark_next;
        id = id_next;
    }
    
    mark_next = __shfl_down_sync(0xFFFFFFFF, mark, 4);
    id_next = __shfl_down_sync(0xFFFFFFFF, id, 4);
    
    if (mark_next >= mark){
        mark = mark_next;
        id = id_next;
    }
    
    mark_next = __shfl_down_sync(0xFFFFFFFF, mark, 2);
    id_next = __shfl_down_sync(0xFFFFFFFF, id, 2);
    
    if (mark_next >= mark){
        mark = mark_next;
        id = id_next;
    }
    
    mark_next = __shfl_down_sync(0xFFFFFFFF, mark, 1);
    id_next = __shfl_down_sync(0xFFFFFFFF, id, 1);
    
    if (mark_next >= mark){
        mark = mark_next;
        id = id_next;
    }
    
    if(lane_id == 0){
        d_reduced_records->assignment_marks[idx] = mark;
        d_reduced_records->student_ids[idx] = id;
    }
    
}

#endif //KERNEL_H