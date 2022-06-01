#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <armadillo>

namespace torque {
namespace gpu {
namespace sparse {

__device__
void index_to_indices (
        unsigned long index,
        const unsigned long * index_table,
        const unsigned long * sorted_index,
        const unsigned long rank,
        unsigned long * indices_output
        ) {

    for(unsigned long i=1; i<=rank; i++) {
        const unsigned long sorted_i = sorted_index[rank - i];
        indices_output[sorted_i] = index / index_table[sorted_i];
        index %= index_table[sorted_i];
    }

}


__global__
void handle_indices(
        const unsigned long * A_indices,
        const unsigned long A_indices_length,
        const unsigned long * B_indices,
        const unsigned long B_indices_length,
        const unsigned long * A_index_table,
        const unsigned long * A_sorted_index,
        const int A_rank,
        const unsigned long * B_index_table,
        const unsigned long * B_sorted_index,
        const int B_rank,
        unsigned long * output
        ) {

    extern __shared__ unsigned long cached_A_table[];
    extern __shared__ unsigned long cached_B_table[];
    extern __shared__ unsigned long cached_A_sort_index[];
    extern __shared__ unsigned long cached_B_sort_index[];

    __shared__ unsigned long A_cached[1024];
    __shared__ unsigned long B_cached[1024];

    __shared__ unsigned long output_cached[1024];

    unsigned long thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long stride = blockDim.x * gridDim.x;

    unsigned long tid = threadIdx.x;

    // copy the indices to shared memory
    if(tid < A_rank) {
         cached_A_table[tid] = A_index_table[tid];
    } else {
        if(tid < 2 * A_rank) {
            cached_A_sort_index[tid - A_rank] = A_sorted_index[tid - A_rank];
        } else {
            if(tid < 2 * A_rank + B_rank) {
                cached_B_table[tid - 2 * A_rank] = B_index_table[tid - 2 * A_rank];
            } else {
                if (tid < 2 * A_rank + 2 * B_rank) {
                    cached_B_table[tid - 2 * A_rank - B_rank] = B_index_table[tid - 2 * A_rank - B_rank];
                }
            }
        }
    }

    __syncthreads();

    B_cached[tid] =



}



}
}
}