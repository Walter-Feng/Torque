#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <iostream>
#ifndef ARMA_ALLOW_FAKE_GCC
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#define MAX_RANK 10
#define BW 32

namespace torque {
namespace gpu {
namespace sparse {

__device__
void index_to_indices (
        int32_t index_i,
        const int32_t * index_table_i,
        const int32_t * sorted_index_i,
        const int32_t rank_i,
        int32_t index_j,
        const int32_t * index_table_j,
        const int32_t * sorted_index_j,
        const int32_t rank_j,
        const int32_t * i_contracting_indices,
        const int32_t * j_contracting_indices,
        const int32_t contracting_ndim,
        const int32_t * i_free_indices,
        const int32_t i_free_indices_length,
        const int32_t * j_free_indices,
        const int32_t j_free_indices_length,
        const int32_t * index_table_out,
        const int32_t out_rank,
        int32_t * indices_output
        ) {

    int32_t i_indices[MAX_RANK];
    int32_t j_indices[MAX_RANK];
    int32_t out_indices[MAX_RANK];

    for(int32_t i=1; i<=rank_i; i++) {
        const int32_t sorted_i = sorted_index_i[BW * (rank_i - i)];
        i_indices[sorted_i] = index_i / index_table_i[BW * sorted_i];
        index_i %= index_table_i[BW * (sorted_i)];
    }

    for(int32_t j=1; j<=rank_j; j++) {
        const int32_t sorted_j = sorted_index_j[BW * (rank_j - j)];
        j_indices[sorted_j] = index_j / index_table_j[BW * (sorted_j)];
        index_j %= index_table_j[BW * (sorted_j)];
    }

    for(int32_t i=0; i<i_free_indices_length; i++) {
        out_indices[i] = i_indices[i_free_indices[i]];
    }

    for(int32_t j=0; j<j_free_indices_length; j++) {
        out_indices[j + i_free_indices_length] = j_indices[j_free_indices[j]];
    }

    int32_t result = 0;

    for(int32_t out=0; out<out_rank; out++) {
        result += index_table_out[out] * out_indices[out];
    }

    bool is_contributing = true;
    for(int32_t i=0; i<contracting_ndim; i++) {
        if(i_indices[i_contracting_indices[i]] != j_indices[j_contracting_indices[i]]) {
            is_contributing = false;
        }
    }

    if(is_contributing) {
        *indices_output = result;
    } else {
        *indices_output = -1;
    }

}


__global__
void handle_indices_kernel(
        const int32_t * A_indices,
        const uint32_t A_indices_length,
        const int32_t * B_indices,
        const uint32_t B_indices_length,
        const int32_t * A_index_table,
        const int32_t * A_sorted_index,
        const uint32_t A_rank,
        const int32_t * B_index_table,
        const int32_t * B_sorted_index,
        const uint32_t B_rank,
        const int32_t * A_contracting_indices,
        const int32_t * B_contracting_indices,
        const uint32_t contracting_ndim,
        const int32_t * A_free_indices,
        const uint32_t A_free_indices_length,
        const int32_t * B_free_indices,
        const uint32_t B_free_indices_length,
        const int32_t * index_table_out,
        const uint32_t out_rank,
        int32_t * output
        ) {

//    __shared__ int32_t cached_A_table[MAX_RANK];
//    __shared__ int32_t cached_B_table[MAX_RANK];
//    __shared__ int32_t cached_A_sort_index[MAX_RANK];
//    __shared__ int32_t cached_B_sort_index[MAX_RANK];
//
//    __shared__ int32_t cached_A_converted_indices[MAX_RANK];
//    __shared__ int32_t cached_B_converted_indices[MAX_RANK];
//
//    __shared__ int32_t cached_i_contracting_indices[MAX_RANK];
//    __shared__ int32_t cached_j_contracting_indices[MAX_RANK];
//
//    __shared__ int32_t cached_i_free_indices[MAX_RANK];
//    __shared__ int32_t cached_j_free_indices[MAX_RANK];
//
//
//
//    __shared__ int32_t cached_result[BW * BW];

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t j = blockIdx.y * blockDim.y + threadIdx.y;


//    int32_t flattened_tid = threadIdx.x + blockDim.x * threadIdx.y;

//    for(int32_t i_cache = 0; i_cache < A_rank; i_cache++) {
//        cached_A_table[i_cache] = A_index_table[i_cache];
//        cached_A_sort_index[i_cache] = A_sorted_index[i_cache];
//    }
//
//    for(int32_t i_cache = 0; i_cache < B_rank; i_cache++) {
//        cached_B_table[i_cache] = B_index_table[i_cache];
//        cached_B_sort_index[i_cache] = B_sorted_index[i_cache];
//    }
//    __syncthreads();

    // copy the indices to shared memory
//    if(flattened_tid < A_rank) {
//         cached_A_table[flattened_tid] = A_index_table[flattened_tid];
//    } else {
//        if(flattened_tid < 2 * A_rank) {
//            cached_A_sort_index[flattened_tid - A_rank] = A_sorted_index[flattened_tid - A_rank];
//        } else {
//            if(flattened_tid < 2 * A_rank + B_rank) {
//                cached_B_table[flattened_tid - 2 * A_rank] = B_index_table[flattened_tid - 2 * A_rank];
//            } else {
//                if (flattened_tid < 2 * A_rank + 2 * B_rank) {
//                    cached_B_sort_index[flattened_tid - 2 * A_rank - B_rank] = B_sorted_index[flattened_tid - 2 * A_rank - B_rank];
//                }
//            }
//        }
//    }

//    __syncthreads();

//    index_to_indices(A_indices[thread_index], )

    if(i < A_indices_length) {
        if(j < B_indices_length) {
            index_to_indices(A_indices[i],
                             A_index_table,
                             A_sorted_index,
                             A_rank,
                             B_indices[j],
                             B_index_table,
                             B_sorted_index,
                             B_rank,
                             A_contracting_indices,
                             B_contracting_indices,
                             contracting_ndim,
                             A_free_indices,
                             A_free_indices_length,
                             B_free_indices,
                             B_free_indices_length,
                             index_table_out,
                             out_rank,
                             output + 32 * j + i);
        }
    }

}

thrust::device_vector<int32_t>  handle_indices(
        const thrust::device_vector<int32_t> & A_indices,
        const thrust::device_vector<int32_t> & B_indices,
        const thrust::device_vector<int32_t> & A_index_table,
        const thrust::device_vector<int32_t> & A_sorted_index,
        const thrust::device_vector<int32_t> & B_index_table,
        const thrust::device_vector<int32_t> & B_sorted_index,
        const thrust::device_vector<int32_t> & A_contracting_indices,
        const thrust::device_vector<int32_t> & B_contracting_indices,
        const thrust::device_vector<int32_t> & A_free_indices,
        const thrust::device_vector<int32_t> & B_free_indices,
        const thrust::device_vector<int32_t> & index_table_out
) {
    assert(A_index_table.size() == A_sorted_index.size());
    assert(B_index_table.size() == B_sorted_index.size());
    assert(A_contracting_indices.size() == B_contracting_indices.size());

    const uint32_t A_indices_length = A_indices.size();
    const uint32_t B_indices_length = B_indices.size();
    const uint32_t A_rank = A_index_table.size();
    const uint32_t B_rank = B_index_table.size();

    const uint32_t contracting_ndim = A_contracting_indices.size();

    dim3 blockSize(32, 32);
    dim3 gridSize(A_indices_length / 32 + 1, B_indices_length / 32 + 1);

    thrust::device_vector<int32_t> output(A_indices.size() * B_indices.size());

    handle_indices_kernel<<<blockSize, gridSize>>>(
            thrust::raw_pointer_cast(A_indices.data()),
            A_indices.size(),
            thrust::raw_pointer_cast(B_indices.data()),
            B_indices.size(),
            thrust::raw_pointer_cast(A_index_table.data()),
            thrust::raw_pointer_cast(A_sorted_index.data()),
            A_rank,
            thrust::raw_pointer_cast(B_index_table.data()),
            thrust::raw_pointer_cast(B_sorted_index.data()),
            B_rank,
            thrust::raw_pointer_cast(A_contracting_indices.data()),
            thrust::raw_pointer_cast(B_contracting_indices.data()),
            contracting_ndim,
            thrust::raw_pointer_cast(A_free_indices.data()),
            A_free_indices.size(),
            thrust::raw_pointer_cast(B_free_indices.data()),
            B_free_indices.size(),
            thrust::raw_pointer_cast(index_table_out.data()),
            index_table_out.size(),
            thrust::raw_pointer_cast(output.data())
    );

    return output;
}


thrust::device_vector<int32_t>  handle_indices(
        const thrust::device_vector<int32_t> & A_indices,
        const thrust::device_vector<int32_t> & B_indices,
        const arma::uvec & A_index_table,
        const arma::uvec & B_index_table,
        const arma::umat & contracting_indices
) {

}


}
}
}
#endif
