#include "block_sparse.cuh"

#define MAX_RANK 10

namespace torque {
namespace gpu {
namespace block_sparse {
    template<typename T, bool reverse>
    __global__
    void
    reshape_kernel(const T * src_data,
                   const int * block_index_tables,
                   const int * blocks_strides,
                   const int * blocks_offsets,
                   int n_block,
                   int n_elem,
                   int rank,
                   const int * dest_index_table,
                   T * dest_data) {

        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < n_elem) {
            int block_index = -1;
            int tensor_index = 0;
            int stacked_index = 0;
            int tmp;

            for (int j = 0; j < n_block; j++) {
                if (i >= blocks_offsets[j]) {
                    block_index += 1;
                } else {
                    block_index += 0;
                }
            }

            int block_offset = blocks_offsets[block_index];
            int tensor_residue = i - block_offset;

            for (int j = 0; j < rank; j++) {

                tmp = tensor_residue / block_index_tables[block_index * rank + rank - j - 1];

                tensor_index += blocks_strides[block_index * rank + rank - j - 1] * tmp;
                stacked_index += dest_index_table[rank - j - 1] * tmp;

                tensor_residue %= block_index_tables[block_index * rank + rank - j - 1];

            }

            tensor_index += block_offset;
            stacked_index += block_index * dest_index_table[rank];

            if constexpr(reverse) {
                dest_data[tensor_index] = src_data[stacked_index];
            } else {
                dest_data[stacked_index] = src_data[tensor_index];
            }
        }

    }

    template
    __global__
    void
    reshape_kernel<float, true>(const float * src_data,
                                const int * block_index_tables,
                                const int * blocks_strides,
                                const int * blocks_offsets,
                                int n_block,
                                int n_elem,
                                int rank,
                                const int * dest_index_table,
                                float * dest_data);

    template
    __global__
    void
    reshape_kernel<double, true>(const double * src_data,
                                 const int * block_index_tables,
                                 const int * blocks_strides,
                                 const int * blocks_offsets,
                                 int n_block,
                                 int n_elem,
                                 int rank,
                                 const int * dest_index_table,
                                 double * dest_data);

    template
    __global__
    void
    reshape_kernel<half, true>(const half * src_data,
                               const int * block_index_tables,
                               const int * blocks_strides,
                               const int * blocks_offsets,
                               int n_block,
                               int n_elem,
                               int rank,
                               const int * dest_index_table,
                               half * dest_data);

    template
    __global__
    void
    reshape_kernel<float, false>(const float * src_data,
                                 const int * block_index_tables,
                                 const int * blocks_strides,
                                 const int * blocks_offsets,
                                 int n_block,
                                 int n_elem,
                                 int rank,
                                 const int * dest_index_table,
                                 float * dest_data);

    template
    __global__
    void
    reshape_kernel<double, false>(const double * src_data,
                                  const int * block_index_tables,
                                  const int * blocks_strides,
                                  const int * blocks_offsets,
                                  int n_block,
                                  int n_elem,
                                  int rank,
                                  const int * dest_index_table,
                                  double * dest_data);

    template
    __global__
    void
    reshape_kernel<half, false>(const half *src_data,
                                const int * block_index_tables,
                                const int * blocks_strides,
                                const int * blocks_offsets,
                                int n_block,
                                int n_elem,
                                int rank,
                                const int * dest_index_table,
                                half * dest_data);
}
}
}
