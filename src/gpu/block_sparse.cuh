#ifndef TORQUE_GPU_BLOCK_SPARSE_CUH
#define TORQUE_GPU_BLOCK_SPARSE_CUH

#include <cutt.h>

#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

template<typename T, bool reverse>
void
reshape(T * dest_data,
        const T * src_data,
        const arma::umat & blocks_dimensions,
        const arma::umat & blocks_strides,
        const arma::uvec & blocks_offsets,
        const arma::uvec & dest_dimensions) {

  const arma::uword n_blocks = blocks_dimensions.n_cols;
  const arma::uword rank = blocks_dimensions.n_rows;

  arma::umat blocks_index_tables(arma::size(blocks_dimensions));

  for (uint32_t i = 0; i < n_blocks; i++) {
    blocks_index_tables.col(i) = torque::util::generate_index_table(
        blocks_dimensions.col(i));
  }

  uint32_t * dev_block_index_tables;
  gpuErrchk(cudaMalloc(&dev_block_index_tables,
                       sizeof(uint32_t) * blocks_index_tables.n_elem));
  util::arma_to_cuda(dev_block_index_tables,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(blocks_index_tables)));

  uint32_t * dev_blocks_strides;
  gpuErrchk(
      cudaMalloc(&dev_blocks_strides,
                 sizeof(uint32_t) * blocks_strides.n_elem));
  util::arma_to_cuda(dev_blocks_strides,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(blocks_strides)));

  uint32_t * dev_block_offsets;
  gpuErrchk(
      cudaMalloc(&dev_block_offsets, sizeof(uint32_t) * blocks_offsets.n_elem));
  util::arma_to_cuda(dev_block_offsets, blocks_offsets);

  const arma::uvec padded_dest_dimensions = arma::join_vert(dest_dimensions,
                                                            arma::uvec{
                                                                n_blocks});

  const arma::uvec dest_index_table = torque::util::generate_index_table(
      padded_dest_dimensions);

  uint32_t * dev_dest_index_table;
  gpuErrchk(
      cudaMalloc(&dev_dest_index_table,
                 sizeof(uint32_t) * dest_index_table.n_elem));
  util::arma_to_cuda(dev_dest_index_table,
                     arma::conv_to<arma::Col<uint32_t>>::from(
                         arma::vectorise(dest_index_table)));

  const arma::uvec n_elem_nest_sum =
      arma::cumsum(arma::prod(blocks_dimensions).t()) -
      arma::prod(blocks_dimensions).t();

  const arma::uword n_elem = arma::sum(arma::prod(blocks_dimensions));

  uint32_t * n_elem_nest_sum_dev;
  gpuErrchk(
      cudaMalloc(&n_elem_nest_sum_dev,
                 sizeof(uint32_t) * n_elem_nest_sum.n_elem));
  util::arma_to_cuda(n_elem_nest_sum_dev, n_elem_nest_sum);

  uint threads_per_block = 1024;
  uint blocks = n_elem / 1024 + 1;

  cudaDeviceSynchronize();

  reshape_kernel<T, reverse><<<blocks, threads_per_block>>>(
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

  cudaDeviceSynchronize();

  gpuErrchk(cudaFree(dev_block_index_tables));
  gpuErrchk(cudaFree(dev_blocks_strides));
  gpuErrchk(cudaFree(dev_block_offsets));
  gpuErrchk(cudaFree(n_elem_nest_sum_dev));
  gpuErrchk(cudaFree(dev_dest_index_table));

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
  contract(cublasHandle_t handle,
           const BlockSparseTensor<T> & tensor,
           const arma::umat & contracting_indices) const {

//            cudaStream_t stream1, stream2;
//
//            cudaStreamCreate(&stream1);
//            cudaStreamCreate(&stream2);

    T * A_copies;
    T * B_copies;
    T * A_transposed_pointer;
    T * B_transposed_pointer;

    const auto permutation_generator =
        [](const arma::uvec & contracting_indices,
           const arma::uword target_rank) -> arma::uvec {

          arma::uvec transposition(target_rank);

          for (int j = 0; j < target_rank; j++) {
            transposition(j) = j;
          }

          transposition.shed_rows(contracting_indices);

          return arma::join_vert(
              arma::join_vert(transposition, contracting_indices),
              arma::uvec{target_rank});
        };


    const arma::uvec this_contracting_indices = contracting_indices.col(0);
    const arma::uvec that_contracting_indices = contracting_indices.col(1);

    const arma::uvec A_permutation =
        permutation_generator(this_contracting_indices, this->rank);

    const arma::uvec B_permutation =
        permutation_generator(that_contracting_indices, tensor.rank);


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
    arma::umat total_begin_points;
    arma::umat total_end_points;

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
        total_begin_points = arma::join_horiz(total_begin_points,
                                              contracting_info.new_begin_points);
        total_end_points = arma::join_horiz(total_end_points,
                                            contracting_info.new_end_points);
        const arma::umat dimension_slice =
            contracting_info.new_end_points
            - contracting_info.new_begin_points
            + arma::ones<arma::umat>(
                arma::size(contracting_info.new_begin_points));

        n_elem_wrt_A_block.push_back(arma::sum(arma::prod(dimension_slice)));
      }
    }

    if (non_trivial_A_block_indices.empty()) {
      return BlockSparseTensor<T>(new_dimension);
    }

    arma::umat new_blocks_dimensions =
        total_end_points - total_begin_points +
        arma::ones<arma::umat>(arma::size(total_begin_points));

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
      gpuErrchk(cudaMalloc(&result_data, sizeof(T)));
    } else {
      gpuErrchk(cudaMalloc(&result_data,
                           sizeof(T) * arma::sum(n_elem_wrt_A_block_in_uvec)));
    }

    T * dot_temp;
    if (result_rank == 0) {
      gpuErrchk(cudaMalloc(&dot_temp, sizeof(T)));
    }


#pragma acc kernels
    {
      for (arma::uword non_trivial_i = 0;
           non_trivial_i < contraction_infos.size(); non_trivial_i++) {

        arma::uword i = non_trivial_A_block_indices[non_trivial_i];

        const torque::block_sparse::ContractionInfo contracting_info = contraction_infos[i];

        const arma::umat A_subblock_rel_begin_points =
            contracting_info.A_begin_points -
            arma::repmat(this->begin_points.col(i), 1,
                         contracting_info.block_indices.n_elem);

        const arma::uvec A_subblock_offsets =
            A_subblock_rel_begin_points.t() * this->index_tables.col(i) +
            this->block_offsets(i);

        const arma::umat A_subblock_end_points = contracting_info.A_end_points;

        const arma::umat A_subblock_dimension =
            A_subblock_end_points
            - contracting_info.A_begin_points
            +
            arma::ones<arma::umat>(arma::size(contracting_info.A_begin_points));

        const arma::uvec B_block_indices = contracting_info.block_indices;
        const arma::uword n_subblocks = B_block_indices.n_elem;
        const arma::umat B_subblock_rel_begin_points =
            contracting_info.B_begin_points -
            tensor.begin_points.cols(B_block_indices);
        const arma::umat B_subblock_end_points = contracting_info.B_end_points;
        const arma::umat B_subblock_dimension =
            B_subblock_end_points
            - contracting_info.B_begin_points
            +
            arma::ones<arma::umat>(arma::size(contracting_info.B_begin_points));

        const arma::uvec B_subblock_offsets =
            arma::sum(B_subblock_rel_begin_points %
                      tensor.index_tables.cols(B_block_indices)).t()
            + tensor.block_offsets.rows(B_block_indices);

        const arma::uvec A_block_max_dimension = arma::max(A_subblock_dimension,
                                                           1);
        const arma::uvec B_block_max_dimension = arma::max(B_subblock_dimension,
                                                           1);

        const arma::uvec padded_A_block_max_dimension =
            arma::join_vert(A_block_max_dimension, arma::uvec{n_subblocks});
        const arma::uvec padded_B_block_max_dimension =
            arma::join_vert(B_block_max_dimension, arma::uvec{n_subblocks});

        arma::uvec A_block_max_dimension_copy = A_block_max_dimension;
        arma::uvec B_block_max_dimension_copy = B_block_max_dimension;

        A_block_max_dimension_copy.shed_rows(this_contracting_indices);
        B_block_max_dimension_copy.shed_rows(that_contracting_indices);


        const arma::uvec dimension_after_multiplication =
            arma::join_vert(A_block_max_dimension_copy,
                            B_block_max_dimension_copy);

        gpuErrchk(cudaMalloc(&A_copies,
                             arma::prod(A_block_max_dimension) * n_subblocks *
                             sizeof(T)));

        block_sparse::reshape<T, false>(
            A_copies,
            this->data,
            A_subblock_dimension,
            arma::repmat(this->index_tables.col(i), 1, n_subblocks),
            A_subblock_offsets,
            A_block_max_dimension
        );

        gpuErrchk(cudaMalloc(&B_copies,
                             arma::prod(B_block_max_dimension) * n_subblocks *
                             sizeof(T)));

        block_sparse::reshape<T, false>(
            B_copies,
            tensor.data,
            B_subblock_dimension,
            tensor.index_tables.cols(B_block_indices),
            B_subblock_offsets,
            B_block_max_dimension
        );

        const arma::uvec A_non_trivial_dimension_in_original_order =
            padded_A_block_max_dimension(
                arma::find(padded_A_block_max_dimension != 1));
        const arma::uvec B_non_trivial_dimension_in_original_order =
            padded_B_block_max_dimension(
                arma::find(padded_B_block_max_dimension != 1));

        const arma::uvec permuted_A_dimension = padded_A_block_max_dimension.rows(
            A_permutation);
        const arma::uvec permuted_B_dimension = padded_B_block_max_dimension.rows(
            B_permutation);

        const arma::uvec A_non_trivial_indices = arma::find(
            permuted_A_dimension != 1);
        const arma::uvec B_non_trivial_indices = arma::find(
            permuted_B_dimension != 1);

        const arma::uvec A_non_trivial_permutation = A_permutation(
            A_non_trivial_indices);
        const arma::uvec B_non_trivial_permutation = B_permutation(
            B_non_trivial_indices);

        const arma::uword A_cutt_rank = A_non_trivial_indices.n_elem;
        const arma::uword B_cutt_rank = B_non_trivial_indices.n_elem;

        std::vector<int> A_dim_in_cutt = std::vector<int>(A_cutt_rank);
        std::vector<int> A_permutation_in_cutt = std::vector<int>(A_cutt_rank);

        assert(A_non_trivial_dimension_in_original_order.n_elem == A_cutt_rank);
        assert(B_non_trivial_dimension_in_original_order.n_elem == B_cutt_rank);

        for (arma::uword j = 0; j < A_cutt_rank; j++) {
          A_dim_in_cutt[j] = A_non_trivial_dimension_in_original_order(j);
          A_permutation_in_cutt[j] = A_non_trivial_permutation(j);
        }

        std::vector<int> B_dim_in_cutt = std::vector<int>(B_cutt_rank);
        std::vector<int> B_permutation_in_cutt = std::vector<int>(B_cutt_rank);

        for (arma::uword j = 0; j < B_cutt_rank; j++) {
          B_dim_in_cutt[j] = B_non_trivial_dimension_in_original_order(j);
          B_permutation_in_cutt[j] = B_non_trivial_permutation(j);
        }

        cuttHandle planA, planB;

        const bool A_is_sorted = A_non_trivial_permutation.is_sorted();
        const bool B_is_sorted = B_non_trivial_permutation.is_sorted();

        if (!A_is_sorted) {
          assert(arma::prod(padded_A_block_max_dimension) != 0);
          gpuErrchk(cudaMalloc(&A_transposed_pointer,
                               arma::prod(padded_A_block_max_dimension) *
                               sizeof(T)));

          cuttCheck(cuttPlanMeasure(&planA, A_cutt_rank,
                                    A_dim_in_cutt.data(),
                                    A_permutation_in_cutt.data(),
                                    sizeof(T), 0, A_copies,
                                    A_transposed_pointer));

          cuttCheck(cuttExecute(planA, A_copies, A_transposed_pointer));

          cuttCheck(cuttDestroy(planA));
          gpuErrchk(cudaFree(A_copies));
        }

        if (!B_is_sorted) {
          assert(arma::prod(padded_B_block_max_dimension) != 0);
          gpuErrchk(cudaMalloc(&B_transposed_pointer,
                               arma::prod(padded_B_block_max_dimension) *
                               sizeof(T)));

          cuttCheck(cuttPlanMeasure(&planB, B_cutt_rank,
                                    B_dim_in_cutt.data(),
                                    B_permutation_in_cutt.data(),
                                    sizeof(T), 0, B_copies,
                                    B_transposed_pointer));


          cuttCheck(cuttExecute(planB, B_copies, B_transposed_pointer));

          cuttCheck(cuttDestroy(planB));
          gpuErrchk(cudaFree(B_copies));
        }

        T * A_ptr = !A_is_sorted ? A_transposed_pointer : A_copies;
        T * B_ptr = !B_is_sorted ? B_transposed_pointer : B_copies;

        if (result_rank > 0) {

          const arma::uword contracting_n_elem = arma::prod(
              A_block_max_dimension(this_contracting_indices));


          const arma::uword A_stride = arma::prod(A_block_max_dimension);
          const arma::uword B_stride = arma::prod(B_block_max_dimension);

          const arma::uword A_leading_dim =
              arma::prod(A_block_max_dimension) / contracting_n_elem;
          const arma::uword B_leading_dim =
              arma::prod(B_block_max_dimension) / contracting_n_elem;

          const arma::uword C_stride = A_leading_dim * B_leading_dim;

          T * out_pointer;
          gpuErrchk(
              cudaMalloc(&out_pointer, C_stride * n_subblocks * sizeof(T)));


          T one = 1;
          T zero = 0;

          if constexpr(std::is_same<T, float>::value) {
            cublasCheck(
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          A_leading_dim,
                                          B_leading_dim, contracting_n_elem,
                                          &one,
                                          A_ptr,
                                          A_leading_dim, A_stride,
                                          B_ptr, B_leading_dim, B_stride,
                                          &zero, out_pointer,
                                          A_leading_dim, C_stride,
                                          n_subblocks));
          } else if constexpr(std::is_same<T, double>::value) {
            cublasCheck(
                cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          A_leading_dim,
                                          B_leading_dim, contracting_n_elem,
                                          &one,
                                          A_ptr,
                                          A_leading_dim, A_stride,
                                          B_ptr, B_leading_dim, B_stride,
                                          &zero, out_pointer,
                                          A_leading_dim, C_stride,
                                          n_subblocks));
          } else if constexpr(std::is_same<T, half>::value) {
            cublasCheck(
                cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                          A_leading_dim,
                                          B_leading_dim, contracting_n_elem,
                                          &one,
                                          A_ptr,
                                          A_leading_dim, A_stride,
                                          B_ptr, B_leading_dim, B_stride,
                                          &zero, out_pointer,
                                          A_leading_dim, C_stride,
                                          n_subblocks));
          }

          A_is_sorted ? cudaFree(A_copies) : cudaFree(A_transposed_pointer);
          B_is_sorted ? cudaFree(B_copies) : cudaFree(B_transposed_pointer);

          const arma::umat new_subblock_dimensions =
              contracting_info.new_end_points
              - contracting_info.new_begin_points
              +
              arma::ones<arma::umat>(
                  arma::size(contracting_info.new_end_points));

          arma::umat new_subblock_index_tables(
              arma::size(new_subblock_dimensions));

          for (arma::uword j = 0; j < n_subblocks; j++) {
            new_subblock_index_tables.col(j) =
                torque::util::generate_index_table(
                    new_subblock_dimensions.col(j));
          }

          arma::umat subblock_offsets =
              arma::cumsum(arma::prod(new_subblock_dimensions)) -
              arma::prod(new_subblock_dimensions);

          T * flattened;
          gpuErrchk(cudaMalloc(&flattened,
                               arma::accu(arma::prod(new_subblock_dimensions)) *
                               sizeof(T)));

          assert(arma::all(dimension_after_multiplication ==
                           arma::max(new_subblock_dimensions, 1)));


          block_sparse::reshape<T, true>(flattened, out_pointer,
                                         new_subblock_dimensions,
                                         new_subblock_index_tables,
                                         subblock_offsets,
                                         dimension_after_multiplication);

          gpuErrchk(cudaFree(out_pointer));

          cudaMemcpy(result_data + offsets_wrt_A_blocks(non_trivial_i),
                     flattened,
                     n_elem_wrt_A_block_in_uvec(non_trivial_i) * sizeof(T),
                     cudaMemcpyDeviceToDevice);

          gpuErrchk(cudaFree(flattened));

        } else { // Full contraction, generating a scalar

          assert(arma::prod(padded_A_block_max_dimension) ==
                 arma::prod(padded_B_block_max_dimension));

          if constexpr(std::is_same<T, float>::value) {
            cublasSdot(handle, arma::prod(padded_A_block_max_dimension), A_ptr,
                       1,
                       B_ptr, 1, dot_temp);
          } else if constexpr(std::is_same<T, double>::value) {
            cublasDdot(handle, arma::prod(padded_A_block_max_dimension), A_ptr,
                       1,
                       B_ptr, 1, dot_temp);
          } else if constexpr(std::is_same<T, half>::value) {
            T one = 1;
            T zero = 0;

            cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1,
                        arma::prod(padded_A_block_max_dimension), &one,
                        A_ptr, arma::prod(padded_A_block_max_dimension), B_ptr,
                        arma::prod(padded_B_block_max_dimension), &zero,
                        dot_temp,
                        1);

          }

          gpuErrchk(cudaFree(A_ptr));
          gpuErrchk(cudaFree(B_ptr));

          block_sparse::add<<<1, 1>>>(result_data, dot_temp);

        }
      }
    };

    if (result_rank == 0) {
      gpuErrchk(cudaFree(dot_temp));
    }

    return {result_data, result_rank, std::move(new_dimension),
            std::move(new_blocks_dimensions),
            std::move(total_begin_points), std::move(total_end_points),
            std::move(new_blocks_n_elem),
            std::move(new_blocks_offsets), std::move(blocks_index_tables)};

//            cudaStreamDestroy(stream1);
//            cudaStreamDestroy(stream2);
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
