#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <iostream>

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "util/space.h"
#include "util/thrust_arma_fusion.cuh"

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
        const uint32_t rank_i,
        int32_t index_j,
        const int32_t * index_table_j,
        const int32_t * sorted_index_j,
        const uint32_t rank_j,
        const int32_t * i_contracting_indices,
        const int32_t * j_contracting_indices,
        const uint32_t contracting_ndim,
        const int32_t * i_free_indices,
        const uint32_t i_free_indices_length,
        const int32_t * j_free_indices,
        const uint32_t j_free_indices_length,
        const int32_t * index_table_out,
        const uint32_t out_rank,
        int32_t * indices_output
        ) {

    int32_t i_indices[MAX_RANK];
    int32_t j_indices[MAX_RANK];
    int32_t out_indices[MAX_RANK];

    for(int32_t i=1; i<=rank_i; i++) {
        const int32_t sorted_i = sorted_index_i[(rank_i - i)];
        i_indices[sorted_i] = index_i / index_table_i[sorted_i];
        index_i %= index_table_i[sorted_i];
    }

    for(int32_t j=1; j<=rank_j; j++) {
        const int32_t sorted_j = sorted_index_j[(rank_j - j)];
        j_indices[sorted_j] = index_j / index_table_j[sorted_j];
        index_j %= index_table_j[sorted_j];
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


    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

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
                             output + A_indices_length * j + i);
        }
    }

}

thrust::device_vector<int32_t> handle_indices(
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





}
}
}

