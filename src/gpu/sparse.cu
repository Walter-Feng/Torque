#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <armadillo>

#define MAX_RANK 10
#define BW 32

namespace torque {
namespace gpu {
namespace sparse {

__device__
void index_to_indices (
        unsigned long index_i,
        const unsigned long * index_table_i,
        const unsigned long * sorted_index_i,
        const unsigned long rank_i,
        unsigned long index_j,
        const unsigned long * index_table_j,
        const unsigned long * sorted_index_j,
        const unsigned long rank_j,
        const unsigned long * i_contracting_indices,
        const unsigned long * j_contracting_indices,
        const unsigned long * contracting_ndim,
        const unsigned long * index_table_out,
        unsigned long * indices_output
        ) {

    unsigned long i_indices[MAX_RANK];
    unsigned long j_indices[MAX_RANK];

    for(unsigned long i=1; i<=rank_i; i++) {
        const unsigned long sorted_i = sorted_index_i[rank_i - i];
        i_indices[sorted_i] = index_i / index_table_i[sorted_i];
        index_i %= index_table_i[sorted_i];
    }

    for(unsigned long j=1; j<=rank_j; j++) {
        const unsigned long sorted_j = sorted_index_j[rank_j - j];
        j_indices[sorted_j] = index_j / index_table_j[sorted_j];
        index_j %= index_table_j[sorted_j];
    }



}


__global__
void handle_indices_kernel(
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
        const unsigned long * index_table_out,
        unsigned long * output
        ) {

    __shared__ unsigned long cached_A_table[BW * MAX_RANK];
    __shared__ unsigned long cached_B_table[BW * MAX_RANK];
    __shared__ unsigned long cached_A_sort_index[BW * MAX_RANK];
    __shared__ unsigned long cached_B_sort_index[BW * MAX_RANK];

    __shared__ unsigned long cached_A_converted_indices[BW * MAX_RANK];
    __shared__ unsigned long cached_B_converted_indices[BW * MAX_RANK];

    __shared__ unsigned long cached_result[BW * BW];

    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long j = blockIdx.y * blockDim.y + threadIdx.y;

    for(unsigned long i_cache = 0; i_cache < A_rank; i_cache++) {
        cached_A_table[BW * threadIdx.x + i_cache] = A_index_table[i_cache];
        cached_A_sort_index[BW * threadIdx.x + i_cache] = A_sorted_index[i_cache];
    }

    for(unsigned long i_cache = 0; i_cache < B_rank; i_cache++) {
        cached_B_table[BW * threadIdx.y + i_cache] = B_index_table[i_cache];
        cached_B_sort_index[BW * threadIdx.y + i_cache] = B_sorted_index[i_cache];
    }

    __syncthreads();

    

//    // copy the indices to shared memory
//    if(tid < A_rank) {
//         cached_A_table[tid] = A_index_table[tid];
//    } else {
//        if(tid < 2 * A_rank) {
//            cached_A_sort_index[tid - A_rank] = A_sorted_index[tid - A_rank];
//        } else {
//            if(tid < 2 * A_rank + B_rank) {
//                cached_B_table[tid - 2 * A_rank] = B_index_table[tid - 2 * A_rank];
//            } else {
//                if (tid < 2 * A_rank + 2 * B_rank) {
//                    cached_B_sort_index[tid - 2 * A_rank - B_rank] = B_sorted_index[tid - 2 * A_rank - B_rank];
//                }
//            }
//        }
//    }

    __syncthreads();

    index_to_indices(A_indices[thread_index], )



}



}
}
}