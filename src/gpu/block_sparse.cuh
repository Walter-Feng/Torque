#ifndef TORQUE_GPU_BLOCK_SPARSE_CUH
#define TORQUE_GPU_BLOCK_SPARSE_CUH

#include <cutt.h>

#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutensor.h>
#include <optional>

#include "tensor/block_sparse.h"
#include "gpu/util/thrust_arma_fusion.cuh"
#include "gpu/util/lib_helper.cuh"


#include "error.h"

#include "util/space.h"

namespace torque {
namespace gpu {
namespace block_sparse {

template<typename T>
__global__ void add(T * a, T * b) //Kernel Definition
{
  *a = *a + *b;
}

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
               T * dest_data);

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
                          T * dest_data);

template<typename T, bool reverse>
void
reshape(T * dest_data,
        const T * src_data,
        const arma::umat & blocks_dimensions,
        const arma::umat & blocks_strides,
        const arma::uvec & blocks_offsets,
        const arma::uvec & dest_dimensions,
        cudaStream_t stream = 0) {

  cudaStream_t sub_stream[5];
  for (int i = 0; i < 5; i++) {
    cudaStreamCreate(sub_stream + i);
  }

  const arma::uword n_blocks = blocks_dimensions.n_cols;
  const arma::uword rank = blocks_dimensions.n_rows;

  arma::umat blocks_index_tables(arma::size(blocks_dimensions));

  for (uint32_t i = 0; i < n_blocks; i++) {
    blocks_index_tables.col(i) = torque::util::generate_index_table(
        blocks_dimensions.col(i));
  }

  uint32_t * dev_block_index_tables;
  gpuErrchk(cudaMallocAsync(&dev_block_index_tables,
                            sizeof(uint32_t) * blocks_index_tables.n_elem,
                            sub_stream[0]));
  util::arma_to_cuda(dev_block_index_tables,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(blocks_index_tables)), sub_stream[0]);

  uint32_t * dev_blocks_strides;
  gpuErrchk(
      cudaMallocAsync(&dev_blocks_strides,
                      sizeof(uint32_t) * blocks_strides.n_elem, sub_stream[1]));
  util::arma_to_cuda(dev_blocks_strides,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(blocks_strides)), sub_stream[1]);

  uint32_t * dev_block_offsets;
  gpuErrchk(
      cudaMallocAsync(&dev_block_offsets,
                      sizeof(uint32_t) * blocks_offsets.n_elem, sub_stream[2]));
  util::arma_to_cuda(dev_block_offsets,
                     arma::conv_to<arma::Col<uint32_t>>::from(blocks_offsets),
                     sub_stream[2]);

  const arma::uvec padded_dest_dimensions = arma::join_vert(dest_dimensions,
                                                            arma::uvec{
                                                                n_blocks});

  const arma::uvec dest_index_table = torque::util::generate_index_table(
      padded_dest_dimensions);

  uint32_t * dev_dest_index_table;
  gpuErrchk(
      cudaMallocAsync(&dev_dest_index_table,
                      sizeof(uint32_t) * dest_index_table.n_elem,
                      sub_stream[3]));
  util::arma_to_cuda(dev_dest_index_table,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(dest_index_table)), sub_stream[3]);

  const arma::Col<uint32_t> n_elem_nest_sum =
      arma::conv_to<arma::Col<uint32_t>>::from(
          arma::cumsum(arma::prod(blocks_dimensions).t()) -
          arma::prod(blocks_dimensions).t());

  const arma::uword n_elem = arma::sum(arma::prod(blocks_dimensions));

  uint32_t * n_elem_nest_sum_dev;
  gpuErrchk(
      cudaMallocAsync(&n_elem_nest_sum_dev,
                      sizeof(uint32_t) * n_elem_nest_sum.n_elem,
                      sub_stream[4]));
  util::arma_to_cuda(n_elem_nest_sum_dev, n_elem_nest_sum, sub_stream[4]);

  uint threads_per_block = 1024;
  uint blocks = n_elem / 1024 + 1;

  cudaStreamSynchronize(sub_stream[0]);
  cudaStreamSynchronize(sub_stream[1]);
  cudaStreamSynchronize(sub_stream[2]);
  cudaStreamSynchronize(sub_stream[3]);
  cudaStreamSynchronize(sub_stream[4]);

  reshape_kernel<T, reverse><<<blocks, threads_per_block, 0, stream>>>(
      src_data,
      dev_block_index_tables,
      dev_blocks_strides,
      dev_block_offsets,
      n_elem_nest_sum_dev,
      n_blocks,
      n_elem,
      rank,
      dev_dest_index_table,
      dest_data
  );

  gpuErrchk(cudaFree(dev_block_index_tables));
  gpuErrchk(cudaFree(dev_blocks_strides));
  gpuErrchk(cudaFree(dev_block_offsets));
  gpuErrchk(cudaFree(n_elem_nest_sum_dev));
  gpuErrchk(cudaFree(dev_dest_index_table));

  for (auto & i: sub_stream) {
    cudaStreamDestroy(i);
  }
}

template<typename T>
void
reshape_with_boost(T * dest_data,
                   const T * src_data,
                   const arma::umat & blocks_dimensions,
                   const arma::umat & boosts,
                   const arma::umat & blocks_strides,
                   const arma::uvec & blocks_offsets,
                   const arma::uvec & dest_dimensions,
                   cudaStream_t stream = 0) {

  cudaStream_t sub_stream[6];
  for (int i = 0; i < 6; i++) {
    cudaStreamCreate(sub_stream + i);
  }

  const arma::uword n_blocks = blocks_dimensions.n_cols;
  const arma::uword rank = blocks_dimensions.n_rows;

  arma::umat blocks_index_tables(arma::size(blocks_dimensions));

  for (uint32_t i = 0; i < n_blocks; i++) {
    blocks_index_tables.col(i) = torque::util::generate_index_table(
        blocks_dimensions.col(i));
  }

  uint32_t * dev_block_index_tables;
  gpuErrchk(cudaMallocAsync(&dev_block_index_tables,
                            sizeof(uint32_t) * blocks_index_tables.n_elem,
                            sub_stream[0]));
  util::arma_to_cuda(dev_block_index_tables,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(blocks_index_tables)), sub_stream[0]);

  uint32_t * dev_blocks_strides;
  gpuErrchk(
      cudaMallocAsync(&dev_blocks_strides,
                      sizeof(uint32_t) * blocks_strides.n_elem, sub_stream[1]));
  util::arma_to_cuda(dev_blocks_strides,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(blocks_strides)), sub_stream[1]);

  const arma::Col<uint32_t> vectorised_boost =
      arma::conv_to<arma::Col<uint32_t>>::from(boosts);

  uint32_t * boost_dev;
  gpuErrchk(cudaMallocAsync(&boost_dev, sizeof(uint32_t) * boosts.n_elem,
                            sub_stream[5]));
  util::arma_to_cuda(boost_dev, vectorised_boost, sub_stream[5]);

  uint32_t * dev_block_offsets;
  gpuErrchk(
      cudaMallocAsync(&dev_block_offsets,
                      sizeof(uint32_t) * blocks_offsets.n_elem, sub_stream[2]));
  util::arma_to_cuda(dev_block_offsets,
                     arma::conv_to<arma::Col<uint32_t>>::from(blocks_offsets),
                     sub_stream[2]);

  const arma::uvec padded_dest_dimensions = arma::join_vert(dest_dimensions,
                                                            arma::uvec{
                                                                n_blocks});

  const arma::uvec dest_index_table = torque::util::generate_index_table(
      padded_dest_dimensions);

  uint32_t * dev_dest_index_table;
  gpuErrchk(
      cudaMallocAsync(&dev_dest_index_table,
                      sizeof(uint32_t) * dest_index_table.n_elem,
                      sub_stream[3]));
  util::arma_to_cuda(dev_dest_index_table,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(dest_index_table)), sub_stream[3]);

  const arma::Col<uint32_t> n_elem_nest_sum =
      arma::conv_to<arma::Col<uint32_t>>::from(
          arma::cumsum(arma::prod(blocks_dimensions).t()) -
          arma::prod(blocks_dimensions).t());

  const arma::uword n_elem = arma::sum(arma::prod(blocks_dimensions));

  uint32_t * n_elem_nest_sum_dev;
  gpuErrchk(
      cudaMallocAsync(&n_elem_nest_sum_dev,
                      sizeof(uint32_t) * n_elem_nest_sum.n_elem,
                      sub_stream[4]));
  util::arma_to_cuda(n_elem_nest_sum_dev, n_elem_nest_sum, sub_stream[4]);

  uint threads_per_block = 1024;
  uint blocks = n_elem / 1024 + 1;

  cudaStreamSynchronize(sub_stream[0]);
  cudaStreamSynchronize(sub_stream[1]);
  cudaStreamSynchronize(sub_stream[2]);
  cudaStreamSynchronize(sub_stream[3]);
  cudaStreamSynchronize(sub_stream[4]);
  cudaStreamSynchronize(sub_stream[5]);

  reshape_kernel_with_boost<T><<<blocks, threads_per_block, 0, stream>>>(
      src_data,
      dev_block_index_tables,
      boost_dev,
      dev_blocks_strides,
      dev_block_offsets,
      n_elem_nest_sum_dev,
      n_blocks,
      n_elem,
      rank,
      dev_dest_index_table,
      dest_data
  );

  gpuErrchk(cudaFree(dev_block_index_tables));
  gpuErrchk(cudaFree(dev_blocks_strides));
  gpuErrchk(cudaFree(dev_block_offsets));
  gpuErrchk(cudaFree(n_elem_nest_sum_dev));
  gpuErrchk(cudaFree(dev_dest_index_table));

  for (int i = 0; i < 6; i++) {
    cudaStreamDestroy(sub_stream[i]);
  }
}
}

/// a tensor object that stores blocks of sub-tensors. The blocks may have intersections.
template<typename T>
class BlockSparseTensor {
public:

  /// The rank of the tensor
  arma::uword rank;

  /// The dimensions at each index of the tensor as a whole.
  arma::uvec dimension;

  /// The dimensions at each index of the tensor for each block. The first index is the leading dimension of the
  /// tensor with stride equal to 1, i.e. difference of 1 for the first index will result in
  /// neighboring address in the data.
  arma::umat blocks_dimension;

  /// Intermediate tables that helps generating the index for the flattened one-dimensional data of each block,
  /// each col stores one index table
  arma::umat index_tables;

  /// The indices of initial element from the blocks. The indices are stored
  /// column-wise, i.e. each column stores one index set for a block.
  /// e.g. for a dense matrix the begin dimensions are (0,0), i.e. rows start from
  /// 0-index row (first row), columns start from 0-index column (first column).
  arma::umat begin_points;

  /// The indices of last non-trivial elements for different blocks.
  /// The indices are stored column-wise,
  /// i.e. each column stores one index set for a block.
  /// e.g. for a dense matrix the end dimensions are (n_rows - 1 , n_cols - 1)
  arma::umat end_points;

  /// The number of elements for each block
  arma::uvec block_n_elem;

  /// The position of first element of the blocks.
  arma::uvec block_offsets;

  inline
  explicit BlockSparseTensor(const arma::uvec & dimension) {
    this->rank = dimension.n_elem;
    this->dimension = dimension;
    gpuErrchk(cudaMalloc(&this->data, sizeof(T)));

    gpuErrchk(cudaMemset(this->data, 0, sizeof(T)));
  }

  inline
  explicit BlockSparseTensor(const T * source_data,
                             const arma::umat & begin_points,
                             const arma::umat & end_points,
                             const arma::uvec & total_dimension,
                             const cudaMemcpyKind kind = cudaMemcpyHostToDevice) {

    rank = total_dimension.n_elem;

    this->dimension = total_dimension;

    if (rank > 0) {
      this->begin_points = begin_points;
      this->end_points = end_points;

      const auto n_blocks = begin_points.n_cols;
      this->blocks_dimension = end_points - begin_points +
                               arma::ones<arma::umat>(arma::size(begin_points));

      this->block_n_elem = arma::prod(this->blocks_dimension).t();
      this->block_offsets = torque::util::nest_sum(this->block_n_elem);

      this->index_tables = arma::umat(arma::size(begin_points));
      for (arma::uword i = 0; i < n_blocks; i++) {
        this->index_tables.col(i) = torque::util::generate_index_table(
            this->blocks_dimension.col(i));
      }

      gpuErrchk(cudaMalloc(&this->data, arma::sum(block_n_elem) * sizeof(T)));

      if (source_data) {
        gpuErrchk(cudaMemcpy(this->data, source_data,
                             sizeof(T) * arma::sum(block_n_elem), kind));
      } else {
        throw Error("Source data not allocated!");
      }
    } else {
      gpuErrchk(cudaMalloc(&this->data, arma::sum(block_n_elem) * sizeof(T)));

      if (source_data) {
        gpuErrchk(cudaMemcpy(this->data, source_data,
                             sizeof(T) * arma::sum(block_n_elem), kind));
      } else {
        throw Error("Source data not allocated!");
      }
    }

  }

  inline
  explicit BlockSparseTensor(const T * source_data,
                             const arma::umat & begin_points,
                             const arma::umat & end_points,
                             const arma::uvec & total_dimension,
                             const arma::umat & index_tables,
                             const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {

    rank = total_dimension.n_elem;

    this->dimension = total_dimension;

    if (rank > 0) {
      this->begin_points = begin_points;
      this->end_points = end_points;

      this->blocks_dimension = end_points - begin_points +
                               arma::ones<arma::umat>(arma::size(begin_points));

      this->block_n_elem = arma::prod(this->blocks_dimension).t();
      this->block_offsets = torque::util::nest_sum(this->block_n_elem);

      this->index_tables = index_tables;

      gpuErrchk(cudaMalloc(&this->data, arma::sum(block_n_elem) * sizeof(T)));

      if (source_data) {
        gpuErrchk(cudaMemcpy(this->data, source_data,
                             sizeof(T) * arma::sum(block_n_elem), kind));
      } else {
        throw Error("Source data not allocated!");
      }
    } else {
      gpuErrchk(cudaMalloc(&this->data, arma::sum(block_n_elem) * sizeof(T)));

      if (source_data) {
        gpuErrchk(cudaMemcpy(this->data, source_data,
                             sizeof(T) * arma::sum(block_n_elem), kind));
      } else {
        throw Error("Source data not allocated!");
      }
    }

  }

  inline
  BlockSparseTensor(T * data,
                    arma::uword rank,
                    arma::uvec && dimension,
                    arma::umat && blocks_dimension,
                    arma::umat && begin_points,
                    arma::umat && end_points,
                    arma::uvec && block_n_elem,
                    arma::uvec && block_offsets,
                    arma::umat && index_tables) {

    this->data = data;
    this->rank = rank;
    this->dimension = std::move(dimension);
    this->blocks_dimension = std::move(blocks_dimension);
    this->begin_points = std::move(begin_points);
    this->end_points = std::move(end_points);
    this->block_n_elem = std::move(block_n_elem);
    this->block_offsets = std::move(block_offsets);
    this->index_tables = std::move(index_tables);

  }

  inline
  BlockSparseTensor(const BlockSparseTensor & tensor) {
    this->rank = tensor.rank;
    this->dimension = tensor.dimension;
    this->blocks_dimension = tensor.blocks_dimension;
    this->begin_points = tensor.begin_points;
    this->end_points = tensor.end_points;
    this->block_n_elem = tensor.block_n_elem;
    this->block_offsets = tensor.block_offsets;
    this->index_tables = tensor.index_tables;

    const arma::uword n_data = arma::sum(arma::prod(this->blocks_dimension));

    gpuErrchk(cudaMalloc(&this->data, n_data * sizeof(T)));

    if (tensor.data) {
      gpuErrchk(cudaMemcpy(this->data, tensor.data, n_data * sizeof(T),
                           cudaMemcpyDeviceToDevice))
    } else {
      throw Error("Source data not allocated!");
    }
  }

  friend void swap(BlockSparseTensor & first, BlockSparseTensor & second) {
    std::swap(first.data, second.data);
    std::swap(first.rank, second.rank);
    std::swap(first.dimension, second.dimension);
    std::swap(first.blocks_dimension, second.blocks_dimension);
    std::swap(first.begin_points, second.begin_points);
    std::swap(first.end_points, second.end_points);
    std::swap(first.block_n_elem, second.block_n_elem);
    std::swap(first.block_offsets, second.block_offsets);
    std::swap(first.index_tables, second.index_tables);
  }


  BlockSparseTensor(BlockSparseTensor && other) noexcept {
    swap(*this, other);
  }

  inline
  BlockSparseTensor & operator=(BlockSparseTensor<T> && other) noexcept {
    swap(*this, other);
    return *this;
  }

  inline
  BlockSparseTensor & operator=(const BlockSparseTensor<T> & other) noexcept {
    BlockSparseTensor temp(other);
    swap(*this, temp);

    return *this;
  }


  inline
  ~BlockSparseTensor() {
    gpuErrchk(cudaFree(this->data));
  }

  ///
  inline
  T to_number() const {
    assert(this->rank == 0);
    T host;
    gpuErrchk(cudaMemcpy(&host, this->data, sizeof(T), cudaMemcpyDeviceToHost));

    return host;
  }

  inline
  void append_block(const T * source_data,
                    const arma::uvec & begin_point,
                    const arma::uvec & end_point,
                    const arma::uvec & index_table,
                    const cudaMemcpyKind kind = cudaMemcpyHostToDevice) {

    this->begin_points = arma::join_horiz(this->begin_points, begin_point);
    this->end_points = arma::join_horiz(this->end_points, end_point);

    const arma::uvec block_dimension =
        end_point - begin_point +
        arma::ones<arma::uvec>(arma::size(begin_point));
    const arma::uword n_elem = arma::prod(block_dimension);

    const arma::uword original_n_elem = arma::sum(this->block_n_elem);

    this->blocks_dimension = arma::join_horiz(this->blocks_dimension,
                                              block_dimension);
    this->block_n_elem = arma::join_vert(this->block_n_elem,
                                         arma::uvec{n_elem});
    this->index_tables = arma::join_horiz(this->index_tables, index_table);
    this->block_offsets = arma::join_vert(this->block_offsets,
                                          arma::uvec{original_n_elem});

    T * new_data;
    gpuErrchk(cudaMalloc(&new_data, arma::sum(this->block_n_elem) * sizeof(T)));
    gpuErrchk(cudaMemcpy(new_data, this->data, original_n_elem * sizeof(T),
                         cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaFree(this->data));
    this->data = new_data;

    gpuErrchk(cudaMemcpy(this->data + original_n_elem, source_data,
                         sizeof(T) * n_elem, kind));
  }

  inline
  void append_blocks(const T * const source_data,
                     const arma::umat & begin_point,
                     const arma::umat & end_point,
                     const arma::umat & index_table,
                     const cudaMemcpyKind kind = cudaMemcpyHostToDevice) {

    this->begin_points = arma::join_horiz(this->begin_points, begin_point);
    this->end_points = arma::join_horiz(this->end_points, end_point);

    const arma::umat block_dimension =
        end_point - begin_point +
        arma::ones<arma::umat>(arma::size(begin_point));
    const arma::uvec n_elem = arma::prod(block_dimension).t();

    const arma::uword original_n_elem = arma::sum(this->block_n_elem);

    this->blocks_dimension = arma::join_horiz(this->blocks_dimension,
                                              block_dimension);
    this->block_n_elem = arma::join_vert(this->block_n_elem, n_elem);
    this->index_tables = arma::join_horiz(this->index_tables, index_table);
    this->block_offsets = arma::join_vert(this->block_offsets,
                                          arma::cumsum(n_elem) - n_elem +
                                          original_n_elem);

    T * new_data;
    gpuErrchk(cudaMalloc(&new_data, arma::sum(this->block_n_elem) * sizeof(T)));
    gpuErrchk(cudaMemcpy(new_data, this->data, original_n_elem * sizeof(T),
                         cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaFree(this->data));
    this->data = new_data;

    gpuErrchk(cudaMemcpy(this->data + original_n_elem, source_data,
                         sizeof(T) * arma::sum(n_elem), kind));
  }

  /// Modify a number in the tensor
  /// \param indices indices for each dimension
  /// \param number the target number (to replace the original one)
  inline
  void modify(const arma::uvec & indices, const T number) {

    if (indices.n_elem != this->rank) {
      throw Error("Rank does not match");
    }

    if (arma::any(indices >= this->dimension)) {
      throw Error("Indices out of boundary");
    }

    if (this->data) {
      const arma::uvec in_range =
          torque::util::in_range(indices, this->begin_points, this->end_points);

      if (in_range.n_elem) {

        const arma::uword block_index = in_range(0);

        const arma::uvec relative_indices =
            indices - this->begin_points.col(block_index);

        gpuErrchk(cudaMemcpy(this->data + block_offsets(block_index)
                             + arma::sum(
                                 relative_indices % this->index_tables.col(block_index)), &number,
                             sizeof(T), cudaMemcpyHostToDevice));

        // all elements at this location in other blocks are set to zero
        for (arma::uword i = 1; i < in_range.n_elem; i++) {
          const T zero = 0;
          const arma::uword block_index_setting_null = in_range(i);
          gpuErrchk(
              cudaMemset(this->data + block_offsets(block_index_setting_null)
                         + arma::sum(relative_indices %
                                     this->index_tables.col(
                                         block_index_setting_null)),
                         0,
                         sizeof(T)));

        }
      } else { // no blocks holding information for this element

        if (number != 0) {
          // append block
          this->append_block(&number, indices, indices,
                             arma::zeros<arma::uvec>(arma::size(indices)));
        }
      }
    } else {
      throw Error("Tensor not initialized");
    }
  }

  /// get the number from tensor with given indces
  /// \param indices indices for each dimension
  inline
  T query(const arma::uvec & indices) const {

    if (indices.n_elem != this->rank) {
      throw Error("Rank does not match");
    }

    if (arma::any(indices >= this->dimension)) {
      throw Error("Indices out of boundary");
    }

    if (this->data) {
      const arma::uvec in_range =
          torque::util::in_range(indices, this->begin_points, this->end_points);

      if (in_range.n_elem) {
        T temp = 0;
        for (arma::uword i = 0; i < in_range.n_elem; i++) {

          const arma::uword block_index = in_range(i);

          const arma::uvec relative_indices =
              indices - this->begin_points.col(block_index);

          T dev_temp;
          const arma::uword displacement = block_offsets(block_index) +
                                           arma::sum(relative_indices %
                                                     this->index_tables.col(
                                                         block_index));

          gpuErrchk(cudaMemcpy((void *) &dev_temp,
                               (void *) (this->data + displacement),
                               sizeof(T), cudaMemcpyDeviceToHost));

          temp += dev_temp;

        }

        return temp;
      } else {
        return 0;
      }

    } else {
      throw Error("Tensor not initialized");
    }
  }

  /// Contraction with another tensor
  /// \param tensor another tensor to be contracted with
  /// \param contracting_indices the corresponding two indices for the dimensions to contract
  /// from two tensors. It should be a (n x 2) matrix, with first col representing "this" tensor.
  BlockSparseTensor<T>
  contract(cutensorHandle_t * cutensor_handle,
           const BlockSparseTensor<T> & tensor,
           const arma::umat & contracting_indices) const {

    const arma::uvec this_contracting_indices = contracting_indices.col(0);
    const arma::uvec that_contracting_indices = contracting_indices.col(1);

    const arma::uvec contract_dimension = this->dimension(
        this_contracting_indices);

    if (!arma::all(
        contract_dimension == tensor.dimension(that_contracting_indices))) {
      throw Error(
          "The dimensions from two tensors to be contracted do not match");
    }

    // Prepare dimension for the new tensor
    arma::uvec this_dimension_copy = this->dimension;
    arma::uvec that_dimension_copy = tensor.dimension;
    this_dimension_copy.shed_rows(this_contracting_indices);
    that_dimension_copy.shed_rows(that_contracting_indices);
    arma::uvec new_dimension = arma::join_vert(this_dimension_copy,
                                               that_dimension_copy);

    const arma::uword result_rank = new_dimension.n_elem;

    std::vector<torque::block_sparse::ContractionInfo> contraction_infos;
    std::vector<arma::uword> non_trivial_A_block_indices;
    std::vector<arma::uword> n_elem_wrt_A_block;
    std::vector<arma::uvec> B_block_max_dimensions;
    std::vector<arma::umat> B_boosts;
    std::vector<arma::uvec> out_block_max_dimensions;
    std::vector<arma::umat> total_begin_points;
    std::vector<arma::umat> total_end_points;

    for (arma::uword i = 0; i < this->block_n_elem.n_elem; i++) {
      const arma::uvec A_begin_point = this->begin_points.col(i);
      const arma::uvec A_end_point = this->end_points.col(i);

      const torque::block_sparse::ContractionInfo contracting_info =
          torque::block_sparse::block_in_range(contracting_indices,
                                               A_begin_point,
                                               A_end_point,
                                               tensor.begin_points,
                                               tensor.end_points);

      if (contracting_info.block_indices.n_elem != 0) {
        contraction_infos.push_back(contracting_info);
        non_trivial_A_block_indices.push_back(i);

        total_begin_points.push_back(contracting_info.new_begin_points);
        total_end_points.push_back(contracting_info.new_end_points);

        const arma::umat dimension_slice =
            contracting_info.new_end_points
            - contracting_info.new_begin_points
            + arma::ones<arma::umat>(
                arma::size(contracting_info.new_begin_points));

        n_elem_wrt_A_block.push_back(arma::sum(arma::prod(dimension_slice)));
        out_block_max_dimensions.push_back(
            arma::join_vert(arma::max(dimension_slice, 1),
                            arma::uvec{contracting_info.block_indices.n_elem}));
        const arma::umat B_dimensions =
            contracting_info.B_end_points - contracting_info.B_begin_points + 1;
        arma::uvec B_max_dimensions = arma::max(B_dimensions, 1);

        B_max_dimensions.rows(
            that_contracting_indices) = this->blocks_dimension.col(i)(
            this_contracting_indices);

        B_block_max_dimensions.push_back(arma::join_vert(B_max_dimensions,
                                                         arma::uvec{
                                                             contracting_info.block_indices.n_elem}));


        arma::umat B_boost(arma::size(B_dimensions), arma::fill::zeros);
        const arma::umat B_begin_points_in_contraction_modes =
            contracting_info.B_begin_points.rows(that_contracting_indices);
        B_boost.rows(that_contracting_indices) =
            B_begin_points_in_contraction_modes.each_col() -
            this->begin_points.col(i)(this_contracting_indices);

        B_boosts.push_back(B_boost);
      }
    }

    if (non_trivial_A_block_indices.empty()) {
      return BlockSparseTensor<T>(new_dimension);
    }

    arma::umat new_blocks_dimensions =
        total_end_points - total_begin_points + 1;

    arma::umat blocks_index_tables(arma::size(new_blocks_dimensions));

    for (int i = 0; i < blocks_index_tables.n_cols; i++) {
      blocks_index_tables.col(i) = torque::util::generate_index_table(
          new_blocks_dimensions.col(i));
    }


    arma::uvec n_elem_wrt_A_block_in_uvec = arma::uvec(n_elem_wrt_A_block);
    arma::uvec offsets_wrt_A_blocks =
        arma::cumsum(n_elem_wrt_A_block_in_uvec) - n_elem_wrt_A_block_in_uvec;
    arma::uvec new_blocks_n_elem = arma::prod(new_blocks_dimensions).t();
    arma::uvec new_blocks_offsets = torque::util::nest_sum(new_blocks_n_elem);

    // temporary variable for dot operation (i.e. result.rank == 0)
    T * result_data;
    if (result_rank == 0) {
      gpuErrchk(cudaMallocAsync(&result_data, sizeof(T), 0));
    } else {
      gpuErrchk(cudaMallocAsync(&result_data,
                                sizeof(T) *
                                arma::sum(n_elem_wrt_A_block_in_uvec, 0)));
    }

    const size_t n_A_blocks = non_trivial_A_block_indices.size();

    cudaStream_t streams[n_A_blocks];

    for (size_t i = 0; i < n_A_blocks; i++) {
      cudaStreamCreate(streams + i);
    }

    T * B_blocks_copies[n_A_blocks];
    T * out_blocks_copies[n_A_blocks];

    for (size_t i = 0; i < n_A_blocks; i++) {
      cudaStreamDestroy(streams[i]);
    }

    for (size_t i = 0; i < n_A_blocks; i++) {
      cudaMallocAsync(&B_blocks_copies[i],
                      arma::prod(B_block_max_dimensions[i]), streams[i]);

      cudaMallocAsync(&out_blocks_copies[i],
                      arma::prod(out_block_max_dimensions[i]), streams[i]);

      const auto & A_index = non_trivial_A_block_indices[i];
      const auto this_dim = arma::conv_to<std::vector<int64_t>>::from(
          this->blocks_dimension.col(A_index));
      const auto this_table = arma::conv_to<std::vector<int64_t>>::from(
          this->index_tables.col(A_index));

      const auto & that_dim = arma::conv_to<std::vector<int64_t>>::from(
          B_block_max_dimensions[i]);
      const auto that_table = arma::conv_to<std::vector<int64_t>>::from(
          torque::util::generate_index_table(B_block_max_dimensions[i]));

      const auto & contraction_info = contraction_infos[i];

      const arma::umat B_blocks_dimension =
          contraction_info.B_begin_points - contraction_info.B_end_points + 1;
      arma::umat B_blocks_strides(arma::size(B_blocks_dimension));
      for (arma::uword j = 0; B_blocks_strides.n_cols; j++) {
        B_blocks_strides.col(j) = torque::util::generate_index_table(
            B_blocks_dimension.col(j));
      }

      const arma::uvec & padded_result_dimension = out_block_max_dimensions[i];

      const auto result_dim =
          arma::conv_to<std::vector<int64_t>>::from(padded_result_dimension);

      const auto result_table = arma::conv_to<std::vector<int64_t>>::from(
          torque::util::generate_index_table(padded_result_dimension));

      const auto compute_type = cutensor_compute_type<T>();
      const auto data_type = cutensor_data_type<T>();

      const arma::uvec B_block_indices = contraction_info.block_indices;
      const arma::umat B_subblock_rel_begin_points =
          contraction_info.B_begin_points -
          tensor.begin_points.cols(B_block_indices);

      const arma::uvec B_subblock_offsets =
          arma::sum(B_subblock_rel_begin_points %
                    tensor.index_tables.cols(B_block_indices)).t()
          + tensor.block_offsets.rows(B_block_indices);

      block_sparse::reshape_with_boost(B_blocks_copies[i], tensor.data,
                                       B_blocks_dimension,
                                       B_boosts[i],
                                       B_blocks_strides,
                                       B_subblock_offsets,
                                       out_block_max_dimensions[i],
                                       streams[i]);

      cutensorTensorDescriptor_t this_descriptor;

      HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle,
                                                &this_descriptor,
                                                this->rank,
                                                this_dim.data(),
                                                this_table.data(),
                                                data_type,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t that_descriptor;

      HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle,
                                                &that_descriptor,
                                                tensor.rank + 1,
                                                that_dim.data(),
                                                that_table.data(),
                                                data_type,
                                                CUTENSOR_OP_IDENTITY));

      cutensorTensorDescriptor_t result_descriptor;
      HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle,
                                                &result_descriptor,
                                                new_dimension.n_elem + 1,
                                                result_dim.data(),
                                                result_table.data(),
                                                data_type,
                                                CUTENSOR_OP_IDENTITY));

      uint32_t this_alignmentRequirement;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle,
                                                   this->data +
                                                   this->block_offsets(A_index),
                                                   &this_descriptor,
                                                   &this_alignmentRequirement));

      uint32_t that_alignmentRequirement;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle,
                                                   B_blocks_copies[i],
                                                   &that_descriptor,
                                                   &that_alignmentRequirement));

      uint32_t result_alignmentRequirement;
      HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle,
                                                   out_blocks_copies[i],
                                                   &result_descriptor,
                                                   &result_alignmentRequirement));

      std::vector<int> total(this->rank + tensor.rank + 1);
      for (int j = 0; j < this->rank + tensor.rank + 1; j++) {
        total[j] = j + 1;
      }
      std::vector<int> this_mode(this->rank);
      std::vector<int> that_mode(tensor.rank);

      memcpy(this_mode.data(), total.data(), sizeof(int) * this->rank);
      memcpy(that_mode.data(), total.data() + this->rank,
             sizeof(int) * (tensor.rank + 1));

      for (int i = 0; i < this_contracting_indices.n_elem; i++) {
        this_mode[this_contracting_indices(i)] = -(i + 1);
        that_mode[that_contracting_indices(i)] = -(i + 1);
        total[this_contracting_indices(i)] = 0;
        total[this->rank + that_contracting_indices(i)] = 0;
      }

      const auto result_mode =
          arma::conv_to<std::vector<int>>::from(
              arma::nonzeros(arma::Col<int>(total)));

      cutensorContractionDescriptor_t desc;

      HANDLE_ERROR(cutensorInitContractionDescriptor(cutensor_handle,
                                                     &desc,
                                                     &this_descriptor,
                                                     this_mode.data(),
                                                     this_alignmentRequirement,
                                                     &that_descriptor,
                                                     that_mode.data(),
                                                     that_alignmentRequirement,
                                                     &result_descriptor,
                                                     result_mode.data(),
                                                     result_alignmentRequirement,
                                                     &result_descriptor,
                                                     result_mode.data(),
                                                     result_alignmentRequirement,
                                                     compute_type));


      cutensorContractionFind_t find;
      HANDLE_ERROR(cutensorInitContractionFind(
          cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));

      // Query workspace
      size_t worksize = 0;
      HANDLE_ERROR(cutensorContractionGetWorkspace(cutensor_handle,
                                                   &desc,
                                                   &find,
                                                   CUTENSOR_WORKSPACE_RECOMMENDED,
                                                   &worksize));

      // Allocate workspace
      void * work[i];
      if (worksize > 0) {
        if (cudaSuccess != cudaMallocAsync(&work[i], worksize,
                                           streams[i])) // This is optional!
        {
          work[i] = nullptr;
          worksize = 0;
        }
      }

      // Create Contraction Plan
      cutensorContractionPlan_t plan;
      HANDLE_ERROR(cutensorInitContractionPlan(cutensor_handle,
                                               &plan,
                                               &desc,
                                               &find,
                                               worksize));

      cutensorStatus_t err;

      T one = 1;
      T zero = 0;

      // Execute the tensor contraction
      err = cutensorContraction(cutensor_handle,
                                &plan,
                                (void *) &one,
                                this->data + this->block_offsets(A_index),
                                B_blocks_copies[i],
                                (void *) &zero,
                                out_blocks_copies[i],
                                out_blocks_copies[i],
                                work, worksize, streams[i]);

      // Check for errors
      if (err != CUTENSOR_STATUS_SUCCESS) {
        printf("ERROR: %s\n", cutensorGetErrorString(err));
      }

      cudaFreeAsync(B_blocks_copies[i], streams[i]);
      block_sparse::reshape<T, true>(result_data + offsets_wrt_A_blocks(i),
                                     out_blocks_copies[i],
                                     new_blocks_dimensions,
                                     new_subblock_index_tables,
                                     subblock_offsets,
                                     dimension_after_multiplication);

    }

    return {result_data, result_rank, std::move(new_dimension),
            std::move(new_blocks_dimensions),
            std::move(total_begin_points), std::move(total_end_points),
            std::move(new_blocks_n_elem),
            std::move(new_blocks_offsets), std::move(blocks_index_tables)};


  }

  /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
  /// This helps keeping the stride of leading dimension equal to 1.
  /// \param permutation the permutation indices
  inline
  BlockSparseTensor<T> hard_transpose(const arma::uvec & permutation) const {

    if (permutation.n_elem != this->rank) {
      throw Error(
          "The number of permutation does not match the rank of tensor");
    }

    const arma::uword n_blocks = this->blocks_dimension.n_cols;

    const arma::uvec max_dimension = arma::max(this->blocks_dimension, 1);
    const arma::uvec new_dimension = this->dimension(permutation);
    const arma::umat new_blocks_dimension = this->blocks_dimension.rows(
        permutation);
    const arma::umat new_begin_points = this->begin_points.rows(permutation);
    const arma::umat new_end_points = this->end_points.rows(permutation);

    T * workspace;
    cudaMalloc(&workspace, arma::prod(max_dimension) * sizeof(T) * n_blocks);
    block_sparse::reshape<T, false>(
        workspace, this->data, this->blocks_dimension,
        this->index_tables, this->block_offsets,
        max_dimension);

    const arma::uvec padded_max_dimension = arma::join_vert(max_dimension,
                                                            arma::uvec{
                                                                n_blocks});
    const arma::uvec padded_permutation = arma::join_vert(permutation,
                                                          arma::uvec{
                                                              this->rank});

    const arma::uvec non_trivial_dimension = arma::find(
        padded_max_dimension != 1);

    const int cutt_rank = non_trivial_dimension.n_elem;

    std::vector<int> dim_in_cutt = std::vector<int>(cutt_rank);
    std::vector<int> permutation_in_cutt = std::vector<int>(cutt_rank);

    for (arma::uword i = 0; i < cutt_rank; i++) {
      dim_in_cutt[i] = padded_max_dimension(non_trivial_dimension(i));
      permutation_in_cutt[i] = padded_permutation(non_trivial_dimension(i));
    }

    T * new_data;
    gpuErrchk(cudaMalloc(&new_data,
                         arma::prod(max_dimension) * sizeof(T) * n_blocks));

    cuttHandle plan;

    cuttCheck(cuttPlanMeasure(&plan, cutt_rank,
                              dim_in_cutt.data(),
                              permutation_in_cutt.data(),
                              sizeof(T), 0, workspace, new_data));

    cuttCheck(cuttExecute(plan, workspace, new_data));

    gpuErrchk(cudaFree(workspace));

    cuttCheck(cuttDestroy(plan));

    arma::umat new_index_tables(arma::size(begin_points));
    for (arma::uword i = 0; i < n_blocks; i++) {
      new_index_tables.col(i) = torque::util::generate_index_table(
          new_blocks_dimension.col(i));
    }

    T * flattened;
    gpuErrchk(cudaMalloc(&flattened,
                         arma::accu(arma::prod(new_blocks_dimension)) *
                         sizeof(T)));

    block_sparse::reshape<T, true>(flattened,
                                   new_data,
                                   new_blocks_dimension,
                                   new_index_tables,
                                   this->block_offsets,
                                   max_dimension(permutation));


    gpuErrchk(cudaFree(new_data));

    const auto a = BlockSparseTensor<T>(flattened,
                                        new_begin_points,
                                        new_end_points,
                                        new_dimension,
                                        cudaMemcpyDeviceToDevice);

    gpuErrchk(cudaFree(flattened));

    return a;
  }

  torque::gpu::BlockSparseTensor<T>
  slice(const arma::uvec & divisor) const {

    const arma::umat sub_blocks = blocks_dimension.each_col() / divisor;

    const arma::uvec divisor_table = torque::util::generate_index_table(
        divisor);

    const arma::uword n_sub_blocks = arma::prod(divisor);

    const arma::umat residue =
        blocks_dimension - sub_blocks.each_col() % divisor;

    if (!residue.is_zero()) {
      throw Error("current slice does not support residues");
    }

    arma::umat new_begin_points;
    arma::umat new_end_points;
    arma::umat new_index_tables;

    for (int i = 0; i < begin_points.n_cols; i++) {

      const arma::uvec begin_point = begin_points.col(i);
      const arma::uvec end_point = end_points.col(i);

      arma::umat new_begin_points_generated(this->rank, n_sub_blocks);
      arma::umat new_end_points_generated(this->rank, n_sub_blocks);

      for (arma::uword j = 0; j < n_sub_blocks; j++) {
        const arma::uvec sub_block_index =
            torque::util::index_to_indices(j, divisor_table);

        new_begin_points_generated.col(j) =
            begin_point + sub_block_index % sub_blocks.col(i);

        new_end_points_generated.col(j) =
            new_begin_points_generated.col(j) + sub_blocks.col(i) - 1;
      }

      new_begin_points = arma::join_horiz(new_begin_points,
                                          new_begin_points_generated);
      new_end_points = arma::join_horiz(new_end_points,
                                        new_end_points_generated);
      new_index_tables =
          arma::join_horiz(new_index_tables,
                           arma::repmat(index_tables.col(i), 1, n_sub_blocks));
    }

    return BlockSparseTensor<T>(this->data, new_begin_points, new_end_points,
                                dimension, cudaMemcpyDeviceToDevice);

  }

protected:
  /// Stores data
  T * data;
};


}
}

#endif //TORQUE_GPU_BLOCK_SPARSE_CUH
