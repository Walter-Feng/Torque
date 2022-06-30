#include "block_sparse.cuh"

#define MAX_RANK 10

__global__
void
reshape(const void ** source_blocks,
        const int ** source_dims,
        const int ** block_index_tables,
        const int ** blocks_strides,
        const int * blocks_table,
        const int n_block,
        const int n_elem,
        const int rank,
        const int * dest_dims,
        void * dest_data) {

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_elem) {
        const uint32_t idx = threadIdx.x;

        int block_index;
        int block_residue;
        int tensor_index;
        int tensor_residue;

        for(int j=0; j<n_block; j++) {

        }
    }

}