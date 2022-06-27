#ifndef TORQUE_GPU_BLOCK_SPARSE_CUH
#define TORQUE_GPU_BLOCK_SPARSE_CUH

#include <cutt.h>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpu/util/thrust_arma_fusion.cuh"
#include <memory>

#include "error.h"

#include "util/space.h"

namespace torque {
namespace gpu {

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
        explicit BlockSparseTensor(const arma::uvec &dimension) {
            this->rank = dimension.n_elem;
            this->dimension = dimension;
            this->data = thrust::device_vector<T>(1);
            this->data[0] = 0;
        }

        inline
        explicit BlockSparseTensor(const T *source_data,
                                   const arma::umat &begin_points,
                                   const arma::umat &end_points,
                                   const arma::uvec &total_dimension) {

            rank = total_dimension.n_elem;

            this->dimension = total_dimension;

            if (rank > 0) {
                this->begin_points = begin_points;
                this->end_points = end_points;

                const auto n_blocks = begin_points.n_cols;
                this->blocks_dimension = end_points - begin_points + arma::ones<arma::umat>(arma::size(begin_points));

                this->block_n_elem = arma::prod(this->blocks_dimension).t();
                this->block_offsets = torque::util::nest_sum(this->block_n_elem);

                this->index_tables = arma::umat(arma::size(begin_points));
                for (arma::uword i = 0; i < n_blocks; i++) {
                    this->index_tables.col(i) = torque::util::generate_index_table(this->blocks_dimension.col(i));
                }

                this->data = thrust::device_vector<T>(arma::sum(block_n_elem));

                if (source_data) {
                    memcpy(this->data.get(), source_data, sizeof(T) * arma::sum(block_n_elem));
                } else {
                    throw Error("Source data not allocated!");
                }
            } else {
                this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::sum(block_n_elem)));

                if (source_data) {
                    memcpy(this->data.get(), source_data, sizeof(T));
                } else {
                    throw Error("Source data not allocated!");
                }
            }

        }

        inline
        explicit BlockSparseTensor(std::unique_ptr<T> &&source_data, const arma::umat &begin_points,
                                   const arma::umat &end_points, const arma::uvec &dimension) {

            if (!source_data) {
                throw Error("Source data not allocated!");
            }

            this->dimension = dimension;

            rank = dimension.n_elem;

            this->begin_points = begin_points;
            this->end_points = end_points;

            const auto n_blocks = begin_points.n_cols;
            this->blocks_dimension = end_points - begin_points + arma::ones<arma::umat>(arma::size(begin_points));

            this->block_n_elem = arma::prod(this->blocks_dimension).t();
            this->block_offsets = torque::util::nest_sum(this->block_n_elem);

            this->index_tables = arma::umat(arma::size(begin_points));
            for (arma::uword i = 0; i < n_blocks; i++) {
                this->index_tables.col(i) = torque::util::generate_index_table(this->blocks_dimension.col(i));
            }

            this->data = std::move(source_data);
        }

        inline
        BlockSparseTensor(const BlockSparseTensor &tensor) {
            this->rank = tensor.rank;
            this->dimension = tensor.dimension;
            this->blocks_dimension = tensor.blocks_dimension;
            this->begin_points = tensor.begin_points;
            this->end_points = tensor.end_points;
            this->block_n_elem = tensor.block_n_elem;
            this->block_offsets = tensor.block_offsets;
            this->index_tables = tensor.index_tables;

            const arma::uword n_data = arma::sum(arma::prod(this->blocks_dimension));

            this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * n_data));

            if (tensor.data) {
                memcpy(this->data.get(), tensor.data.get(), sizeof(T) * n_data);
            } else {
                throw Error("Source data not allocated!");
            }
        }

        inline
        ~BlockSparseTensor() {
            this->data.release();
        }

        ///
        inline
        T to_number() const {
            assert(this->rank == 0);
            return *(this->data.get());

        }

        /// Initialization of the data, with proper memory allocation and memory copy
        inline
        void initialize(const T *source_data) {
            if (!this->data) {
                throw Error("data not allocated!");
            }
            if (source_data) {
                memcpy(this->data.get(), source_data, sizeof(T) * arma::prod(dimension));
            } else {
                throw Error("Source data not allocated!");
            }
        }

        inline
        void append_block(const T *source_data,
                          const arma::uvec &begin_point,
                          const arma::uvec &end_point,
                          const arma::uvec &index_table) {

            this->begin_points = arma::join_horiz(this->begin_points, begin_point);
            this->end_points = arma::join_horiz(this->end_points, end_point);

            const arma::uvec block_dimension =
                    end_point - begin_point + arma::ones<arma::uvec>(arma::size(begin_point));
            const arma::uword n_elem = arma::prod(block_dimension);

            const arma::uword original_n_elem = arma::sum(this->block_n_elem);

            this->blocks_dimension = arma::join_horiz(this->blocks_dimension, block_dimension);
            this->block_n_elem = arma::join_vert(this->block_n_elem, arma::uvec{n_elem});
            this->index_tables = arma::join_horiz(this->index_tables, index_table);
            this->block_offsets = arma::join_vert(this->block_offsets, arma::uvec{original_n_elem});

            T *ptr_new = (T *) realloc(this->data.get(),
                                       arma::sum(this->block_n_elem)
                                       * sizeof(T));

            this->data.release();
            this->data.reset(ptr_new);

            memcpy(this->data.get() + original_n_elem, source_data, n_elem * sizeof(T));
        }

        /// Modify a number in the tensor
        /// \param indices indices for each dimension
        /// \param number the target number (to replace the original one)
        inline
        void modify(const arma::uvec &indices, const T number) {

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

                    this->data.get()[block_offsets(block_index)
                                     + arma::sum(relative_indices % this->index_tables.col(block_index))] = number;

                    // all elements at this location in other blocks are set to zero
                    for (arma::uword i = 1; i < in_range.n_elem; i++) {
                        const arma::uword block_index_setting_null = in_range(i);
                        this->data.get()[block_offsets(block_index_setting_null)
                                         + arma::sum(relative_indices %
                                                     this->index_tables.col(block_index_setting_null))]
                                = 0;
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
        T query(const arma::uvec &indices) const {

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

                        temp +=
                                this->data.get()[block_offsets(block_index) +
                                                 arma::sum(relative_indices % this->index_tables.col(block_index))];
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
        template<typename U>
        BlockSparseTensor<std::common_type_t<T, U>>
        contract(const BlockSparseTensor<U> &tensor, const arma::umat &contracting_indices) const {


        }

        /// Transposition of the tensors according to the permutation, without changing original data
        /// \param permutation the permutation indices
        inline
        void transpose(const arma::uvec &permutation) {

            if (permutation.n_elem != rank) {
                throw Error("The number of permutation does not match the rank of tensor");
            }

            this->index_tables = this->index_tables.rows(permutation);
            this->dimension = this->dimension(permutation);
            this->blocks_dimension = this->blocks_dimension(permutation);
            this->begin_points = this->begin_points.rows(permutation);
            this->end_points = this->end_points.rows(permutation);

        }

        /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
        /// This helps keeping the stride of leading dimension equal to 1.
        /// \param permutation the permutation indices
        inline
        BlockSparseTensor<T> hard_transpose(const arma::uvec &permutation) const {

            if (permutation.n_elem != rank) {
                throw Error("The number of permutation does not match the rank of tensor");
            }

            const arma::uvec new_dimension = this->dimension(permutation);
            const arma::umat new_blocks_dimension = this->blocks_dimension.rows(permutation);
            const arma::umat new_begin_points = this->begin_points.rows(permutation);
            const arma::umat new_end_points = this->end_points.rows(permutation);

            auto result = BlockSparseTensor<T>(new_dimension);

            for (arma::uword i = 0; i < new_begin_points.n_cols; i++) {
                const arma::uvec new_table = torque::util::generate_index_table(new_blocks_dimension.col(i));
                const arma::uword n_elem = arma::prod(new_blocks_dimension.col(i));

                const arma::uvec old_table_sort_index = arma::sort_index(this->index_tables.col(i));

                T *block_data = (T *) malloc(n_elem * sizeof(T));
                for (arma::uword j = 0; j < n_elem; j++) {
                    const arma::uvec old_indices = torque::util::index_to_indices(j, this->index_tables.col(i),
                                                                          old_table_sort_index);
                    const arma::uvec new_indices = old_indices(permutation);

                    block_data[arma::sum(new_indices % new_table)] =
                            this->data.get()[arma::sum(old_indices % this->index_tables.col(i)) +
                                             this->block_offsets(i)];
                }

                result.append_block(block_data, new_begin_points.col(i), new_end_points.col(i), new_table);
            }

            return result;

        }

    protected:
        /// Stores data
        thrust::device_vector<T> data;
    };


}
}

#endif //TORQUE_GPU_BLOCK_SPARSE_CUH
