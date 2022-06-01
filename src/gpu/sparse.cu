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
        long long index,
        const long long * index_table,
        const long long * sorted_index,
        const long long rank,
        long long * indices_output
        ) {

    for(long long i=1; i<=rank; i++) {
        const long long sorted_i = sorted_index[rank - i];
        indices_output[sorted_i] = index / index_table[sorted_i];
        index %= index_table[sorted_i];
    }

}


__global__
void handle_indices(
        const thrust::device_vector<long long> & A_indices,
        const thrust::device_vector<long long> & B_indices,
        const thrust::device_vector<long long> & A_index_table,
        const thrust::device_vector<long long> & A_sorted_index,
        const thrust::device_vector<long long> & B_index_table,
        const thrust::device_vector<long long> & B_sorted_index,
        thrust::device_vector<long long> & output
        ) {

    extern __shared__ long long cached_A_table[];
    extern __shared__ long long cached_B_table[];
    extern __shared__ long long cached_A_sort_index[];
    extern __shared__ long long cached_B_sort_index[];

    __shared__ long long A_cached[1024];
    __shared__ long long B_cached[1024];

    __shared__ long long output_cached[1024];

    long long thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    if(threadIdx.x < A_indices.size()) {
         
    }


}



}
}
}