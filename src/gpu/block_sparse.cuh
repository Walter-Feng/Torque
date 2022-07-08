#ifndef TORQUE_GPU_BLOCK_SPARSE_CUH
#define TORQUE_GPU_BLOCK_SPARSE_CUH

#include <cutt.h>

#define ARMA_ALLOW_FAKE_GCC

#define DEBUG(x) do { printf("shit bug marker %d \n", x); } while (0) ;

#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <optional>

#include "tensor/block_sparse.h"
#include "gpu/util/thrust_arma_fusion.cuh"
#include "gpu/util/lib_helper.cuh"
#include <memory>

#include "error.h"

#include "util/space.h"

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
                   T * dest_data);

    template<typename T, bool reverse>
    thrust::device_vector<T>
    reshape(const thrust::device_vector<T> & src_data,
            const arma::umat & blocks_dimensions,
            const arma::umat & blocks_strides,
            const arma::uvec & blocks_offsets,
            const arma::uvec & dest_dimensions) {

        const arma::uword n_blocks = blocks_dimensions.n_cols;
        const arma::uword rank = blocks_dimensions.n_rows;

        arma::umat blocks_index_tables(arma::size(blocks_dimensions));

        for(int i=0; i<n_blocks; i++) {
            blocks_index_tables.col(i) = torque::util::generate_index_table(blocks_dimensions.col(i));
        }

        thrust::device_vector<int> dev_block_index_tables =
                util::arma_to_thrust_device<int>(arma::conv_to<arma::Col<int>>::from(arma::vectorise(blocks_index_tables)));

        const auto dev_blocks_strides =
                util::arma_to_thrust_device<int>(arma::conv_to<arma::Col<int>>::from(arma::vectorise(blocks_strides)));

        const auto blocks_offsets_in_thrust =
                util::arma_to_thrust_device<int>(blocks_offsets);

        const arma::uvec padded_dest_dimensions = arma::join_vert(dest_dimensions, arma::uvec{n_blocks});

        const arma::uvec dest_index_table = torque::util::generate_index_table(padded_dest_dimensions);
        const thrust::device_vector<int> dest_index_table_in_thrust = util::arma_to_thrust_device<int>(dest_index_table);

        thrust::device_vector<T> dest_data(arma::prod(padded_dest_dimensions));

        const arma::uvec n_elem_nest_sum = arma::cumsum(arma::prod(blocks_dimensions).t());
        const arma::uword n_elem = arma::sum(arma::prod(blocks_dimensions));

        const thrust::device_vector<int> n_elem_nest_sum_in_thrust = util::arma_to_thrust_device<int>(n_elem_nest_sum);

        dim3 blockSize(256);
        dim3 gridSize(n_elem / 256 + 1);

        reshape_kernel<T, reverse><<<blockSize, gridSize>>>(
                thrust::raw_pointer_cast(src_data.data()),
                thrust::raw_pointer_cast(dev_block_index_tables.data()),
                thrust::raw_pointer_cast(dev_blocks_strides.data()),
                thrust::raw_pointer_cast(blocks_offsets_in_thrust.data()),
                (int) n_blocks,
                (int) n_elem,
                (int) rank,
                thrust::raw_pointer_cast(dest_index_table_in_thrust.data()),
                thrust::raw_pointer_cast(dest_data.data())
        );

        return dest_data;
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
                    thrust::copy(source_data, source_data + arma::sum(block_n_elem), this->data.begin());
                } else {
                    throw Error("Source data not allocated!");
                }
            } else {
                this->data = thrust::device_vector<T>(arma::sum(block_n_elem));

                if (source_data) {
                    this->data[0] = source_data[0];
                } else {
                    throw Error("Source data not allocated!");
                }
            }

        }

        inline
        explicit BlockSparseTensor(thrust::device_vector<T> &&source_data, const arma::umat &begin_points,
                                   const arma::umat &end_points, const arma::uvec &dimension) {

            if (source_data.empty()) {
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

            this->data = thrust::device_vector<T>(n_data);

            if (tensor.data) {
                thrust::copy(tensor.data.begin(), tensor.data.end(), this->data.begin());
            } else {
                throw Error("Source data not allocated!");
            }
        }

        BlockSparseTensor(BlockSparseTensor &&tensor)  noexcept :
        rank(std::move(tensor.rank)),
        dimension(std::move(tensor.dimension)),
        blocks_dimension(std::move(tensor.blocks_dimension)),
        begin_points(std::move(tensor.begin_points)),
        end_points(std::move(tensor.end_points)),
        block_n_elem(std::move(tensor.block_n_elem)),
        block_offsets(std::move(tensor.block_offsets)),
        index_tables(std::move(tensor.index_tables)),
        data(std::move(tensor.data)) {}

        inline
        BlockSparseTensor& operator=(BlockSparseTensor<T>&& other)  noexcept {
            rank = std::move(other.rank);
            dimension = std::move(other.dimension),
            blocks_dimension = std::move(other.blocks_dimension);
            begin_points = std::move(other.begin_points);
            end_points = std::move(other.end_points);
            block_n_elem = std::move(other.block_n_elem);
            block_offsets = std::move(other.block_offsets);
            index_tables = std::move(other.index_tables);
            data = std::move(other.data);

            return *this;
        }

        ///
        inline
        T to_number() const {
            assert(this->rank == 0);
            return this->data[0];

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

            this->data.resize(arma::sum(this->block_n_elem) * sizeof(T));

            thrust::copy(source_data, source_data + n_elem, this->data.begin() + original_n_elem);
        }

        inline
        void append_blocks(const thrust::device_vector<T> source_data,
                           const arma::umat &begin_point,
                           const arma::umat &end_point,
                           const arma::umat &index_table) {

            this->begin_points = arma::join_horiz(this->begin_points, begin_point);
            this->end_points = arma::join_horiz(this->end_points, end_point);

            const arma::umat block_dimension =
                    end_point - begin_point + arma::ones<arma::umat>(arma::size(begin_point));
            const arma::uvec n_elem = arma::prod(block_dimension).t();

            const arma::uword original_n_elem = arma::sum(this->block_n_elem);

            this->blocks_dimension = arma::join_horiz(this->blocks_dimension, block_dimension);
            this->block_n_elem = arma::join_vert(this->block_n_elem, n_elem);
            this->index_tables = arma::join_horiz(this->index_tables, index_table);
            this->block_offsets = arma::join_vert(this->block_offsets, arma::cumsum(n_elem) - n_elem);

            this->data.resize(arma::sum(this->block_n_elem) * sizeof(T));

            thrust::copy(source_data.begin(), source_data.end(), this->data.begin() + original_n_elem);
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

            if (!this->data.empty()) {
                const arma::uvec in_range =
                        torque::util::in_range(indices, this->begin_points, this->end_points);

                if (in_range.n_elem) {

                    const arma::uword block_index = in_range(0);

                    const arma::uvec relative_indices =
                            indices - this->begin_points.col(block_index);

                    this->data[block_offsets(block_index)
                               + arma::sum(relative_indices % this->index_tables.col(block_index))] = number;

                    // all elements at this location in other blocks are set to zero
                    for (arma::uword i = 1; i < in_range.n_elem; i++) {
                        const arma::uword block_index_setting_null = in_range(i);
                        this->data[block_offsets(block_index_setting_null)
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

            if (!this->data.empty()) {
                const arma::uvec in_range =
                        torque::util::in_range(indices, this->begin_points, this->end_points);
                if (in_range.n_elem) {
                    T temp = 0;
                    for (arma::uword i = 0; i < in_range.n_elem; i++) {

                        const arma::uword block_index = in_range(i);

                        const arma::uvec relative_indices =
                                indices - this->begin_points.col(block_index);

                        temp +=
                                this->data[block_offsets(block_index) +
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
        BlockSparseTensor<T>
        contract(const BlockSparseTensor<T> &tensor, const arma::umat &contracting_indices) const {

//            cudaStream_t stream1, stream2;
//
//            cudaStreamCreate(&stream1);
//            cudaStreamCreate(&stream2);

            const arma::uvec this_contracting_indices = contracting_indices.col(0);
            const arma::uvec that_contracting_indices = contracting_indices.col(1);

            const arma::uvec contract_dimension = this->dimension(this_contracting_indices);

            if(!arma::all(contract_dimension == tensor.dimension(that_contracting_indices))) {
                throw Error("The dimensions from two tensors to be contracted do not match");
            }

            // Prepare dimension for the new tensor
            arma::uvec this_dimension_copy = this->dimension;
            arma::uvec that_dimension_copy = tensor.dimension;
            this_dimension_copy.shed_rows(this_contracting_indices);
            that_dimension_copy.shed_rows(that_contracting_indices);
            const arma::uvec new_dimension = arma::join_vert(this_dimension_copy, that_dimension_copy);

            const arma::uword result_rank = new_dimension.n_elem;

            auto result = BlockSparseTensor<T>(new_dimension);

            for(arma::uword i=0; i<this->block_n_elem.n_elem; i++) {

                const arma::uvec A_begin_point = this->begin_points.col(i);
                const arma::uvec A_begin_point_in_contracting_dimension = A_begin_point.rows(this_contracting_indices);
                const arma::uvec A_end_point = this->end_points.col(i);

                const torque::block_sparse::ContractionInfo contracting_info =
                        torque::block_sparse::block_in_range(contracting_indices,
                                                             A_begin_point,
                                                             A_end_point,
                                                             tensor.begin_points,
                                                             tensor.end_points);

                if(contracting_info.block_indices.empty()) {
                    continue;
                }

                const arma::umat A_subblock_rel_begin_points = contracting_info.A_begin_points -
                        arma::repmat(this->begin_points.col(i), 1, contracting_info.block_indices.n_elem);

                const arma::uvec A_subblock_offsets =
                        A_subblock_rel_begin_points.t() * this->index_tables.col(i) + this->block_offsets(i);

                const arma::umat A_subblock_end_points = contracting_info.A_end_points;

                const arma::umat A_subblock_dimension =
                        A_subblock_end_points
                        - contracting_info.A_begin_points
                        + arma::ones<arma::umat>(arma::size(contracting_info.A_begin_points));

                const arma::uvec B_block_indices = contracting_info.block_indices;
                const arma::uword n_subblocks = B_block_indices.n_elem;
                const arma::umat B_subblock_rel_begin_points =
                        contracting_info.B_begin_points - tensor.begin_points.cols(B_block_indices);
                const arma::umat B_subblock_end_points = contracting_info.B_end_points;
                const arma::umat B_subblock_dimension =
                        B_subblock_end_points
                        - contracting_info.B_begin_points
                        + arma::ones<arma::umat>(arma::size(contracting_info.B_begin_points));

                const arma::uvec B_subblock_offsets =
                        arma::sum(B_subblock_rel_begin_points % tensor.index_tables.cols(B_block_indices)).t()
                        + tensor.block_offsets.rows(B_block_indices);

                const arma::uvec A_block_max_dimension = arma::max(A_subblock_dimension, 1);
                const arma::uvec B_block_max_dimension = arma::max(B_subblock_dimension, 1);

                const arma::uvec padded_A_block_max_dimension =
                        arma::join_vert(A_block_max_dimension, arma::uvec{n_subblocks});
                const arma::uvec padded_B_block_max_dimension =
                        arma::join_vert(B_block_max_dimension, arma::uvec{n_subblocks});

                const arma::uvec A_non_trivial_dimension = arma::find(padded_A_block_max_dimension != 1);
                const arma::uvec B_non_trivial_dimension = arma::find(padded_B_block_max_dimension != 1);


                arma::uvec A_block_max_dimension_copy = A_block_max_dimension;
                arma::uvec B_block_max_dimension_copy = B_block_max_dimension;

                A_block_max_dimension_copy.shed_rows(this_contracting_indices);
                B_block_max_dimension_copy.shed_rows(that_contracting_indices);


                const arma::uvec dimension_after_multiplication =
                        arma::join_vert(A_block_max_dimension_copy, B_block_max_dimension_copy);

                A_subblock_dimension.print("A_subblock_dimension");
                A_subblock_offsets.print("A_subblock_offsets");
                padded_A_block_max_dimension.print("padded_A_block_max_dimension");
                B_subblock_dimension.print("B_subblock_dimension");
                B_subblock_offsets.print("B_subblock_offsets");
                padded_B_block_max_dimension.print("padded_A_block_max_dimension");

                thrust::device_vector<T> A_copies = block_sparse::reshape<T, false>(
                        this->data,
                        A_subblock_dimension,
                        arma::repmat(this->index_tables.col(i), 1, n_subblocks),
                        A_subblock_offsets,
                        padded_A_block_max_dimension
                        );

                thrust::device_vector<T> B_copies = block_sparse::reshape<T, false>(
                        tensor.data,
                        B_subblock_dimension,
                        tensor.index_tables.cols(B_block_indices),
                        B_subblock_offsets,
                        padded_B_block_max_dimension
                );

                const int A_cutt_rank = A_non_trivial_dimension.n_elem;
                const int B_cutt_rank = B_non_trivial_dimension.n_elem;

                std::vector<int> A_dim_in_cutt = std::vector<int>(A_cutt_rank);
                std::vector<int> A_permutation_in_cutt = std::vector<int>(A_cutt_rank);

                const auto permutation_generator =
                        [](const arma::uvec & contracting_indices, const arma::uword target_rank) -> arma::uvec {

                            arma::uvec transposition(target_rank);

                            for(int j=0; j<target_rank; j++) {
                                transposition(j) = j;
                            }

                            transposition.shed_rows(contracting_indices);

                            return arma::join_vert(arma::join_vert(transposition, contracting_indices), arma::uvec{target_rank});
                        };

                const arma::uvec A_permutation_generated = permutation_generator(this_contracting_indices, this->rank);
                const arma::uvec B_permutation_generated = permutation_generator(that_contracting_indices, tensor.rank);

                const arma::uvec A_permutation =
                        permutation_generator(this_contracting_indices, this->rank)(A_non_trivial_dimension);
                const arma::uvec B_permutation =
                        permutation_generator(that_contracting_indices, tensor.rank)(B_non_trivial_dimension);

                for(arma::uword j=0; j<A_cutt_rank; j++) {
                    A_dim_in_cutt[j] = padded_A_block_max_dimension(A_non_trivial_dimension(j));
                    A_permutation_in_cutt[j] = A_permutation(j);
                }

                std::vector<int> B_dim_in_cutt = std::vector<int>(B_cutt_rank);
                std::vector<int> B_permutation_in_cutt = std::vector<int>(B_cutt_rank);

                for(arma::uword j=0; j<B_cutt_rank; j++) {
                    B_dim_in_cutt[j] = padded_B_block_max_dimension(B_non_trivial_dimension(j));
                    B_permutation_in_cutt[j] = B_permutation(j);
                }

                DEBUG(4)

                std::optional<thrust::device_vector<T>> A_transposed = std::nullopt;
                std::optional<thrust::device_vector<T>> B_transposed = std::nullopt;

                cuttHandle planA, planB;

                if(!A_permutation.is_sorted()) {
                    A_transposed = {thrust::device_vector<T>(arma::prod(padded_A_block_max_dimension))};
                    cuttCheck(cuttPlan(&planA, A_cutt_rank,
                                       A_dim_in_cutt.data(),
                                       A_permutation_in_cutt.data(),
                                       sizeof(T), 0));

                    cuttCheck(cuttExecute(planA,
                                          (void *) const_cast<T *>(thrust::raw_pointer_cast(A_copies.data())),
                                          (void *) thrust::raw_pointer_cast(A_transposed.value().data())));

                    A_copies.clear();
                    A_copies.shrink_to_fit();
                }

                if(!B_permutation.is_sorted()) {
                    B_transposed = {thrust::device_vector<T>(arma::prod(padded_B_block_max_dimension))};
                    cuttCheck(cuttPlan(&planB, B_cutt_rank,
                                       B_dim_in_cutt.data(),
                                       B_permutation_in_cutt.data(),
                                       sizeof(T), 0));

                    cuttCheck(cuttExecute(planB,
                                          (void *) const_cast<T *>(thrust::raw_pointer_cast(B_copies.data())),
                                          (void *) thrust::raw_pointer_cast(B_transposed.value().data())));

                    B_copies.clear();
                    B_copies.shrink_to_fit();
                }

                DEBUG(5)

                std::cout << "A_copies: " << std::endl;
                for(int j=0; j<A_copies.size(); j++) {
                    std::cout << A_copies[j] << " ";
                }
                std::cout << std::endl;

                const T * A_ptr =
                        A_transposed.has_value() ?
                        thrust::raw_pointer_cast(A_transposed.value().data()) :
                        thrust::raw_pointer_cast(A_copies.data());

                std::cout << "B_copies: " << std::endl;
                for(int j=0; j<B_copies.size(); j++) {
                    std::cout << B_copies[j] << " ";
                }
                std::cout << std::endl;

                const T * B_ptr =
                        B_transposed.has_value() ?
                        thrust::raw_pointer_cast(B_transposed.value().data()) :
                        thrust::raw_pointer_cast(B_copies.data());

                cublasHandle_t handle;
                cublasCreate(&handle);

                DEBUG(6)

                if(result_rank > 0) {

                    const arma::uword contracting_n_elem = arma::prod(A_subblock_dimension(this_contracting_indices));
                    const arma::uword this_leading_dim = arma::prod(A_block_max_dimension) / contracting_n_elem;
                    const arma::uword that_leading_dim = arma::prod(B_block_max_dimension) / contracting_n_elem;

                    thrust::device_vector<T> out(this_leading_dim * that_leading_dim * n_subblocks);
                    T * out_pointer = thrust::raw_pointer_cast(out.data());

                    const T * A_array[n_subblocks];
                    const T * B_array[n_subblocks];
                    T * C_array[n_subblocks];

                    for(int j=0; j<n_subblocks; j++) {
                        A_array[j] = A_ptr + j * arma::prod(A_block_max_dimension);
                        B_array[j] = B_ptr + j * arma::prod(B_block_max_dimension);
                        C_array[j] = out_pointer + j * this_leading_dim * that_leading_dim;
                    }

                    T one = 1;
                    T zero = 0;

                    if constexpr(std::is_same<T, float>::value) {
                        cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, this_leading_dim,
                                           that_leading_dim, contracting_n_elem, &one, A_array,
                                           this_leading_dim, B_array, that_leading_dim, &zero, C_array,
                                           this_leading_dim, n_subblocks);
                    } else if constexpr(std::is_same<T, double>::value) {
                        cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, this_leading_dim,
                                           that_leading_dim, contracting_n_elem, &one, A_array,
                                           this_leading_dim, B_array, that_leading_dim, &zero, C_array,
                                           this_leading_dim, n_subblocks);
                    } else if constexpr(std::is_same<T, half>::value) {
                        cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, this_leading_dim,
                                           that_leading_dim, contracting_n_elem, &one, A_array,
                                           this_leading_dim, B_array, that_leading_dim, &zero, C_array,
                                           this_leading_dim, n_subblocks);
                    }

                    const arma::umat new_subblock_dimensions =
                            contracting_info.new_end_points
                            - contracting_info.new_begin_points
                            + arma::ones<arma::umat>(arma::size(contracting_info.new_end_points));

                    arma::umat new_subblock_index_tables(arma::size(new_subblock_dimensions));

                    for(arma::uword j=0; j<n_subblocks; j++) {
                        new_subblock_index_tables.col(j) =
                                torque::util::generate_index_table(new_subblock_dimensions.col(j));
                    }

                    arma::umat subblock_offsets =
                            arma::cumsum(arma::prod(new_subblock_dimensions)) - arma::prod(new_subblock_dimensions);

                    thrust::device_vector<T> flattened(arma::accu(arma::prod(new_subblock_dimensions)));

                    assert(arma::all(dimension_after_multiplication == arma::max(new_subblock_dimensions, 1).t()));

                    block_sparse::reshape<T, true>(out, new_subblock_dimensions,
                                                   new_subblock_index_tables,
                                                   subblock_offsets,
                                                   dimension_after_multiplication);

                    result.append_blocks(out,
                                         contracting_info.new_begin_points,
                                         contracting_info.new_end_points,
                                         new_subblock_index_tables);

                } else { // Full contraction, generating a scalar

                    assert(arma::prod(padded_A_block_max_dimension) == arma::prod(padded_B_block_max_dimension));

                    T dot_per_block;
                    if constexpr(std::is_same<T, float>::value) {
                        cublasSdot(handle, arma::prod(padded_A_block_max_dimension), A_ptr, 1, B_ptr, 1, &dot_per_block);
                    } else if constexpr(std::is_same<T, double>::value) {
                        cublasDdot(handle, arma::prod(padded_A_block_max_dimension), A_ptr, 1, B_ptr, 1, &dot_per_block);
                    } else if constexpr(std::is_same<T, half>::value){
                        T * dot_dev;
                        T one = 1;
                        T zero = 0;
                        cudaMalloc(&dot_dev, sizeof(T));
                        cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1,
                                    arma::prod(padded_A_block_max_dimension), &one,
                                    A_ptr, arma::prod(padded_A_block_max_dimension), B_ptr,
                                    arma::prod(padded_B_block_max_dimension), &zero, dot_dev, 1);

                        cudaMemcpy(dot_per_block, dot_dev, sizeof(T), cudaMemcpyDeviceToHost);
                    }

                    result.data[0] += dot_per_block;
                }

            }

            return result;

//            cudaStreamDestroy(stream1);
//            cudaStreamDestroy(stream2);
        }

        /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
        /// This helps keeping the stride of leading dimension equal to 1.
        /// \param permutation the permutation indices
        inline
        BlockSparseTensor<T> hard_transpose(const arma::uvec &permutation) const {

            if (permutation.n_elem != this->rank) {
                throw Error("The number of permutation does not match the rank of tensor");
            }

            const arma::uword n_blocks = this->blocks_dimension.n_cols;

            const arma::uvec max_dimension = arma::max(this->blocks_dimension, 1);
            const arma::uvec new_dimension = this->dimension(permutation);
            const arma::umat new_blocks_dimension = this->blocks_dimension.rows(permutation);
            const arma::umat new_begin_points = this->begin_points.rows(permutation);
            const arma::umat new_end_points = this->end_points.rows(permutation);

            thrust::device_vector<T> workspace = block_sparse::reshape<T, false>(
                    this->data, this->blocks_dimension,
                    this->index_tables, this->block_offsets,
                    max_dimension);

            const arma::uvec padded_max_dimension = arma::join_vert(max_dimension, arma::uvec{n_blocks});
            const arma::uvec padded_permutation = arma::join_vert(permutation, arma::uvec{this->rank});

            const arma::uvec non_trivial_dimension = arma::find(padded_max_dimension != 1);

            const int cutt_rank = non_trivial_dimension.n_elem;

            std::vector<int> dim_in_cutt = std::vector<int>(cutt_rank);
            std::vector<int> permutation_in_cutt = std::vector<int>(cutt_rank);

            for(arma::uword i=0; i<cutt_rank; i++) {
                dim_in_cutt[i] = padded_max_dimension(non_trivial_dimension(i));
                permutation_in_cutt[i] = padded_permutation(non_trivial_dimension(i));
            }

            auto new_data = thrust::device_vector<T>(arma::prod(padded_max_dimension));


            cuttHandle plan;

            cuttCheck(cuttPlan(&plan, cutt_rank,
                               dim_in_cutt.data(),
                               permutation_in_cutt.data(),
                               sizeof(T), 0));

            cuttCheck(cuttExecute(plan,
                                  (void *) const_cast<T *>(thrust::raw_pointer_cast(workspace.data())),
                                  (void *) thrust::raw_pointer_cast(new_data.data())));

            // empty the vector
            workspace.clear();

            // deallocate any capacity which may currently be associated with vec
            workspace.shrink_to_fit();

            cuttCheck(cuttDestroy(plan));

            arma::umat new_index_tables(arma::size(begin_points));
            for (arma::uword i = 0; i < n_blocks; i++) {
                new_index_tables.col(i) = torque::util::generate_index_table(new_blocks_dimension.col(i));
            }

            thrust::device_vector<T> flattened =
                    block_sparse::reshape<T, true>(new_data,
                                          new_blocks_dimension,
                                          new_index_tables,
                                          this->block_offsets,
                                          max_dimension(permutation));

            return BlockSparseTensor<T>(std::move(flattened), new_begin_points, new_end_points, new_dimension);
        }

    protected:
        /// Stores data
        thrust::device_vector<T> data;
    };


}
}

#endif //TORQUE_GPU_BLOCK_SPARSE_CUH
