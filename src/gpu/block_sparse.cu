#include "block_sparse.cuh"

#define MAX_RANK 10

namespace torque {
namespace gpu {
namespace block_sparse {

template<typename T, bool reverse>
__global__
void
reshape_kernel(const T * src_data,
               const uint32_t * block_index_tables,
               const uint32_t * blocks_strides,
               const uint32_t * blocks_offsets,
               const uint32_t * n_elem_nest_sum,
               uint32_t n_block,
               uint32_t n_elem,
               uint32_t rank,
               const uint32_t * dest_index_table,
               T * dest_data) {

  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_elem) {
    uint32_t block_index = 0;
    uint32_t tensor_index = 0;
    uint32_t stacked_index = 0;
    uint32_t tmp;

    for (uint32_t j = 1; j < n_block; j++) {
      if (i >= n_elem_nest_sum[j]) {
        block_index += 1;
      } else {
        block_index += 0;
      }
    }

    uint32_t block_offset = blocks_offsets[block_index];
    uint32_t tensor_residue = i - n_elem_nest_sum[block_index];

    for (uint32_t j = 0; j < rank; j++) {

      tmp = tensor_residue /
            block_index_tables[block_index * rank + rank - j - 1];

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
                            const uint32_t * block_index_tables,
                            const uint32_t * blocks_strides,
                            const uint32_t * blocks_offsets,
                            const uint32_t * n_elem_nest_sum,
                            uint32_t n_block,
                            uint32_t n_elem,
                            uint32_t rank,
                            const uint32_t * dest_index_table,
                            float * dest_data);

template
__global__
void
reshape_kernel<double, true>(const double * src_data,
                             const uint32_t * block_index_tables,
                             const uint32_t * blocks_strides,
                             const uint32_t * blocks_offsets,
                             const uint32_t * n_elem_nest_sum,
                             uint32_t n_block,
                             uint32_t n_elem,
                             uint32_t rank,
                             const uint32_t * dest_index_table,
                             double * dest_data);

template
__global__
void
reshape_kernel<half, true>(const half * src_data,
                           const uint32_t * block_index_tables,
                           const uint32_t * blocks_strides,
                           const uint32_t * blocks_offsets,
                           const uint32_t * n_elem_nest_sum,
                           uint32_t n_block,
                           uint32_t n_elem,
                           uint32_t rank,
                           const uint32_t * dest_index_table,
                           half * dest_data);

template
__global__
void
reshape_kernel<float, false>(const float * src_data,
                             const uint32_t * block_index_tables,
                             const uint32_t * blocks_strides,
                             const uint32_t * blocks_offsets,
                             const uint32_t * n_elem_nest_sum,
                             uint32_t n_block,
                             uint32_t n_elem,
                             uint32_t rank,
                             const uint32_t * dest_index_table,
                             float * dest_data);

template
__global__
void
reshape_kernel<double, false>(const double * src_data,
                              const uint32_t * block_index_tables,
                              const uint32_t * blocks_strides,
                              const uint32_t * blocks_offsets,
                              const uint32_t * n_elem_nest_sum,
                              uint32_t n_block,
                              uint32_t n_elem,
                              uint32_t rank,
                              const uint32_t * dest_index_table,
                              double * dest_data);

template
__global__
void
reshape_kernel<half, false>(const half * src_data,
                            const uint32_t * block_index_tables,
                            const uint32_t * blocks_strides,
                            const uint32_t * blocks_offsets,
                            const uint32_t * n_elem_nest_sum,
                            uint32_t n_block,
                            uint32_t n_elem,
                            uint32_t rank,
                            const uint32_t * dest_index_table,
                            half * dest_data);

template<typename T>
__global__
void
reshape_kernel_with_boost(const T * src_data,
                          const uint32_t * block_index_tables,
                          const uint32_t * column_boost,
                          const uint32_t * blocks_strides,
                          const uint32_t * blocks_offsets,
                          const uint32_t * n_elem_nest_sum,
                          uint32_t n_block,
                          uint32_t n_elem,
                          uint32_t rank,
                          const uint32_t * dest_index_table,
                          T * dest_data) {

  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_elem) {
    uint32_t block_index = 0;
    uint32_t tensor_index = 0;
    uint32_t stacked_index = 0;
    uint32_t tmp;

    for (uint32_t j = 1; j < n_block; j++) {
      if (i >= n_elem_nest_sum[j]) {
        block_index += 1;
      } else {
        block_index += 0;
      }
    }

    uint32_t block_offset = blocks_offsets[block_index];
    uint32_t tensor_residue = i - n_elem_nest_sum[block_index];

    for (uint32_t j = 0; j < rank; j++) {

      tmp = tensor_residue /
            block_index_tables[block_index * rank + rank - j - 1] +
            column_boost[block_index * rank + j];

      tensor_index += blocks_strides[block_index * rank + rank - j - 1] * tmp;
      stacked_index += dest_index_table[rank - j - 1] * tmp;

      tensor_residue %= block_index_tables[block_index * rank + rank - j - 1];

    }

    tensor_index += block_offset;
    stacked_index += block_index * dest_index_table[rank];


    dest_data[stacked_index] = src_data[tensor_index];

  }

}

template
__global__
void
reshape_kernel_with_boost<float>(const float * src_data,
                                 const uint32_t * block_index_tables,
                                 const uint32_t * column_boost,
                                 const uint32_t * blocks_strides,
                                 const uint32_t * blocks_offsets,
                                 const uint32_t * n_elem_nest_sum,
                                 uint32_t n_block,
                                 uint32_t n_elem,
                                 uint32_t rank,
                                 const uint32_t * dest_index_table,
                                 float * dest_data);

template
__global__
void
reshape_kernel_with_boost<double>(const double * src_data,
                                  const uint32_t * block_index_tables,
                                  const uint32_t * column_boost,
                                  const uint32_t * blocks_strides,
                                  const uint32_t * blocks_offsets,
                                  const uint32_t * n_elem_nest_sum,
                                  uint32_t n_block,
                                  uint32_t n_elem,
                                  uint32_t rank,
                                  const uint32_t * dest_index_table,
                                  double * dest_data);

template
__global__
void
reshape_kernel_with_boost<half>(const half * src_data,
                                const uint32_t * block_index_tables,
                                const uint32_t * column_boost,
                                const uint32_t * blocks_strides,
                                const uint32_t * blocks_offsets,
                                const uint32_t * n_elem_nest_sum,
                                uint32_t n_block,
                                uint32_t n_elem,
                                uint32_t rank,
                                const uint32_t * dest_index_table,
                                half * dest_data);

}
}
}
