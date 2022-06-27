#ifndef TORQUE_GPU_BLOCK_SPARSE_CUH
#define TORQUE_GPU_BLOCK_SPARSE_CUH

#include <cutt.h>
#include <armadillo>
#include <memory>

#include "error.h"

#include "util/space.h"

namespace torque {

    template<typename T>
    arma::Mat<T> sort_into_matrix(const T * block,
                                  const arma::uvec & dimension,
                                  const arma::uvec & table,
                                  const arma::uvec & col_indices) {

        const arma::uvec old_table_sort_index = arma::sort_index(table);

        arma::uvec new_dimension = dimension;
        const arma::uvec dimension_to_col = new_dimension.rows(col_indices);

        new_dimension.shed_rows(col_indices);

        new_dimension = arma::join_vert(new_dimension, dimension_to_col);

        const arma::uword col_n_dim = col_indices.n_elem;
        const arma::uword row_n_dim = dimension.n_elem - col_n_dim;


        const arma::uvec new_table = util::generate_index_table(new_dimension);
        const arma::uword n_row =
                row_n_dim ?
                arma::prod(new_dimension.rows(0, row_n_dim - 1)) :
                1;

        const arma::uword n_col = arma::prod(new_dimension) / n_row;

        const arma::uvec col_indices_sort_index = arma::sort_index(col_indices);
        const arma::uvec sorted_col_indices = col_indices(col_indices_sort_index);

        arma::Mat<T> result(n_row, n_col);

        for(arma::uword j=0; j<result.n_elem; j++) {
            const arma::uvec new_indices = util::index_to_indices(j, new_table);
            arma::uvec old_indices = new_indices;

            const arma::uvec to_col_indices = old_indices.rows(row_n_dim,
                                                               row_n_dim + col_n_dim - 1);
            old_indices.shed_rows(row_n_dim, row_n_dim + col_n_dim - 1);

            for(arma::uword k=0; k<col_indices.n_elem; k++) {
                old_indices.insert_rows(sorted_col_indices(k), 1);
            }

            for(arma::uword k=0; k<col_indices.n_elem; k++) {
                old_indices(col_indices(k)) = new_indices(row_n_dim + k);
            }

            const arma::uvec intermediate_index = new_indices % new_table;

            const arma::uword i_row =
                    row_n_dim ?
                    arma::sum(intermediate_index.rows(0, row_n_dim - 1)) :
                    0;
            const arma::uword i_col =
                    arma::sum(intermediate_index.rows(row_n_dim, dimension.n_elem - 1))
                    / n_row;

            result(i_row, i_col) = block[arma::sum(old_indices % table)];
        }

        return result;

    }

    struct
    ContractionInfo {
        arma::uvec block_indices; // The selection of tensor blocks with non-trivial contribution
        arma::umat new_begin_points; // The begin points of new blocks from contraction
        arma::umat new_end_points; // The end points of new blocks from contraction

        arma::umat contraction_begin_points; // Starting point of the reduced subblock for contraction
        arma::umat contraction_end_points; // End point of the reduced subblock for contraction
        arma::umat contraction_tables; // index table for the contraction that helps iterating over the contracting elements

        arma::umat A_begin_points;
        arma::umat A_end_points;
        arma::umat B_begin_points;
        arma::umat B_end_points;
    };

    ContractionInfo
    block_in_range(const arma::umat & contracting_indices,
                   const arma::uvec & A_begin_point,
                   const arma::uvec & A_end_point,
                   const arma::umat & B_begin_points,
                   const arma::umat & B_end_points) {

        const arma::uword B_n_blocks = B_begin_points.n_cols;
        const arma::uvec A_contracting_indices = contracting_indices.col(0);
        const arma::uvec B_contracting_indices = contracting_indices.col(1);

        // assert A block and B block have consistent number of subblocks and rank
        assert(A_begin_point.n_rows == A_end_point.n_rows);
        assert(B_begin_points.n_rows == B_end_points.n_rows);
        assert(B_begin_points.n_cols == B_end_points.n_cols);

        // The intervals of the non-trivial contribution from the blocks are min(end) - max(begin)
        arma::umat max_begin_indices_in_contracting_dimension(arma::size(contracting_indices.n_rows,
                                                                         B_n_blocks), arma::fill::zeros);

        arma::umat min_end_indices_in_contracting_dimension(arma::size(contracting_indices.n_rows,
                                                                       B_n_blocks), arma::fill::zeros);

        const arma::uvec A_sort_index = arma::sort_index(A_contracting_indices);
        const arma::uvec B_sort_index = arma::sort_index(B_contracting_indices);

        const arma::uvec sorted_A_contracting_indices = A_contracting_indices(A_sort_index);
        const arma::uvec sorted_B_contracting_indices = B_contracting_indices(B_sort_index);

        // Check whether it has non-trivial intervals for each block
        arma::Col<int> true_false_list(B_n_blocks, arma::fill::zeros);

        for(arma::uword i=0; i<B_n_blocks; i++) {
            const arma::uvec i_begin_point = B_begin_points.col(i); // sub-block from B list
            const arma::uvec i_end_point = B_end_points.col(i);

            const arma::uvec max_begin_indices = arma::max(A_begin_point.rows(A_contracting_indices),
                                                           i_begin_point.rows(B_contracting_indices));
            const arma::uvec min_end_indices = arma::min(A_end_point.rows(A_contracting_indices),
                                                         i_end_point.rows(B_contracting_indices));

            if(arma::all(max_begin_indices <= min_end_indices)) {
                true_false_list(i) = 1;

                max_begin_indices_in_contracting_dimension.col(i) = max_begin_indices;
                min_end_indices_in_contracting_dimension.col(i) = min_end_indices;
            }
        }

        const arma::uvec non_trivial_block_index = arma::find(true_false_list);

        if(non_trivial_block_index.n_elem) {

            arma::uvec new_begin_point_from_A = A_begin_point;
            new_begin_point_from_A.shed_rows(A_contracting_indices);

            arma::umat new_begin_points_from_B = B_begin_points.cols(non_trivial_block_index);
            new_begin_points_from_B.shed_rows(B_contracting_indices);

            const arma::uword new_rank = new_begin_point_from_A.n_elem + new_begin_points_from_B.n_rows;

            const arma::umat new_begin_points =
                    new_rank ?
                    arma::join_vert(arma::repmat(new_begin_point_from_A, 1,
                                                 non_trivial_block_index.n_elem),new_begin_points_from_B) :
                    arma::umat{};

            arma::umat new_begin_points_for_A =
                    arma::repmat(new_begin_point_from_A, 1, non_trivial_block_index.n_elem);

            arma::umat new_begin_points_for_B = new_begin_points_from_B;


            for(arma::uword k=0; k<A_contracting_indices.n_elem; k++) {
                new_begin_points_for_A.insert_rows(sorted_A_contracting_indices(k), 1);
                new_begin_points_for_B.insert_rows(sorted_B_contracting_indices(k), 1);
            }

            const arma::umat non_trivial_max_begin_indices =
                    max_begin_indices_in_contracting_dimension.cols(non_trivial_block_index);
            const arma::umat non_trivial_min_end_indices =
                    min_end_indices_in_contracting_dimension.cols(non_trivial_block_index);

            for(arma::uword k=0; k<A_contracting_indices.n_elem; k++) {
                new_begin_points_for_A.row(A_contracting_indices(k)) =
                        non_trivial_max_begin_indices.row(k);
                new_begin_points_for_B.row(B_contracting_indices(k)) =
                        non_trivial_max_begin_indices.row(k);
            }

            arma::uvec new_end_point_from_A = A_end_point;
            new_end_point_from_A.shed_rows(A_contracting_indices);

            arma::umat new_end_points_from_B = B_end_points.cols(non_trivial_block_index);
            new_end_points_from_B.shed_rows(B_contracting_indices);

            const arma::umat new_end_points =
                    new_rank ?
                    arma::join_vert(arma::repmat(new_end_point_from_A, 1,
                                                 non_trivial_block_index.n_elem), new_end_points_from_B) :
                    arma::umat{};

            arma::umat contracting_tables(arma::size(max_begin_indices_in_contracting_dimension));

            for(arma::uword i=0; i<non_trivial_block_index.n_elem; i++) {
                contracting_tables.col(i) = util::generate_index_table(
                        non_trivial_min_end_indices.col(i)
                        - non_trivial_max_begin_indices.col(i)
                        + arma::ones<arma::uvec>(contracting_indices.n_rows));

            }

            arma::umat new_end_points_for_A =
                    arma::repmat(new_end_point_from_A, 1, non_trivial_block_index.n_elem);

            arma::umat new_end_points_for_B = new_end_points_from_B;


            for(arma::uword k=0; k<A_contracting_indices.n_elem; k++) {
                new_end_points_for_A.insert_rows(sorted_A_contracting_indices(k), 1);
                new_end_points_for_B.insert_rows(sorted_B_contracting_indices(k), 1);
            }

            for(arma::uword k=0; k<A_contracting_indices.n_elem; k++) {
                new_end_points_for_A.row(A_contracting_indices(k)) =
                        non_trivial_min_end_indices.row(k);
                new_end_points_for_B.row(B_contracting_indices(k)) =
                        non_trivial_min_end_indices.row(k);
            }

            return {non_trivial_block_index,
                    new_begin_points,
                    new_end_points,
                    non_trivial_max_begin_indices,
                    non_trivial_min_end_indices,
                    contracting_tables,
                    new_begin_points_for_A,
                    new_end_points_for_A,
                    new_begin_points_for_B,
                    new_end_points_for_B
            };

        } else {
            return {arma::uvec{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{},
                    arma::umat{}};
        }



    }
/// a tensor object that stores blocks of sub-tensors. The blocks may have intersections.
    template<typename T>
    class BlockSparseTensor
    {
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
            this->data = std::unique_ptr<T>((T *) malloc(sizeof(T)));
            memset(this->data.get(), 0, sizeof(T));
        }

        inline
        explicit BlockSparseTensor(const T * source_data,
                                   const arma::umat & begin_points,
                                   const arma::umat & end_points,
                                   const arma::uvec & total_dimension) {

            rank = total_dimension.n_elem;

            this->dimension = total_dimension;

            if(rank > 0) {
                this->begin_points = begin_points;
                this->end_points = end_points;

                const auto n_blocks = begin_points.n_cols;
                this->blocks_dimension = end_points - begin_points + arma::ones<arma::umat>(arma::size(begin_points));

                this->block_n_elem = arma::prod(this->blocks_dimension).t();
                this->block_offsets = util::nest_sum(this->block_n_elem);

                this->index_tables = arma::umat(arma::size(begin_points));
                for(arma::uword i=0; i<n_blocks; i++) {
                    this->index_tables.col(i) = util::generate_index_table(this->blocks_dimension.col(i));
                }

                this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::sum(block_n_elem)));

                if(source_data) {
                    memcpy(this->data.get(), source_data, sizeof(T) * arma::sum(block_n_elem));
                } else {
                    throw Error("Source data not allocated!");
                }
            } else {
                this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::sum(block_n_elem)));

                if(source_data) {
                    memcpy(this->data.get(), source_data, sizeof(T));
                } else {
                    throw Error("Source data not allocated!");
                }
            }

        }

        inline
        explicit BlockSparseTensor(std::unique_ptr<T> && source_data, const arma::umat & begin_points,
                                   const arma::umat & end_points, const arma::uvec & dimension) {

            if(!source_data) {
                throw Error("Source data not allocated!");
            }

            this->dimension = dimension;

            rank = dimension.n_elem;

            this->begin_points = begin_points;
            this->end_points = end_points;

            const auto n_blocks = begin_points.n_cols;
            this->blocks_dimension = end_points - begin_points + arma::ones<arma::umat>(arma::size(begin_points));

            this->block_n_elem = arma::prod(this->blocks_dimension).t();
            this->block_offsets = util::nest_sum(this->block_n_elem);

            this->index_tables = arma::umat(arma::size(begin_points));
            for(arma::uword i=0; i<n_blocks; i++) {
                this->index_tables.col(i) = util::generate_index_table(this->blocks_dimension.col(i));
            }

            this->data = std::move(source_data);
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

            this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * n_data));

            if(tensor.data) {
                memcpy(this->data.get(), tensor.data.get(), sizeof(T) * n_data);
            } else {
                throw Error("Source data not allocated!");
            }
        }

        inline
        ~BlockSparseTensor(){
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
        void initialize(const T * source_data) {
            if(!this->data) {
                throw Error("data not allocated!");
            }
            if(source_data) {
                memcpy(this->data.get(), source_data, sizeof(T) * arma::prod(dimension));
            } else {
                throw Error("Source data not allocated!");
            }
        }

        inline
        void append_block(const T * source_data,
                          const arma::uvec & begin_point,
                          const arma::uvec & end_point,
                          const arma::uvec & index_table) {

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

            T * ptr_new = (T *)realloc(this->data.get(),
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
        void modify(const arma::uvec & indices, const T number) {

            if(indices.n_elem != this->rank) {
                throw Error("Rank does not match");
            }

            if(arma::any(indices >= this->dimension)) {
                throw Error("Indices out of boundary");
            }

            if(this->data) {
                const arma::uvec in_range =
                        util::in_range(indices, this->begin_points, this->end_points);

                if(in_range.n_elem) {

                    const arma::uword block_index = in_range(0);

                    const arma::uvec relative_indices =
                            indices - this->begin_points.col(block_index);

                    this->data.get()[block_offsets(block_index)
                                     + arma::sum(relative_indices % this->index_tables.col(block_index))] = number;

                    // all elements at this location in other blocks are set to zero
                    for(arma::uword i=1; i<in_range.n_elem; i++) {
                        const arma::uword block_index_setting_null = in_range(i);
                        this->data.get()[block_offsets(block_index_setting_null)
                                         + arma::sum(relative_indices %
                                                     this->index_tables.col(block_index_setting_null))]
                                = 0;
                    }
                } else { // no blocks holding information for this element

                    if(number != 0) {
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

            if(indices.n_elem != this->rank) {
                throw Error("Rank does not match");
            }

            if(arma::any(indices >= this->dimension)) {
                throw Error("Indices out of boundary");
            }

            if(this->data) {
                const arma::uvec in_range =
                        util::in_range(indices, this->begin_points, this->end_points);
                if(in_range.n_elem) {
                    T temp = 0;
                    for(arma::uword i=0; i<in_range.n_elem; i++) {

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
        contract(const BlockSparseTensor<U> & tensor, const arma::umat & contracting_indices) const {
            const arma::uvec this_contracting_indices = contracting_indices.col(0);
            const arma::uvec that_contracting_indices = contracting_indices.col(1);

            // It might be that the contracting indices may not be sorted, for example the matrix inner product
            // of two matrices (A, B) = sum( A % B^T ), where % is the element-wise multiplication
            // To restore the shed indices, we need the sorted contracting indices
            const arma::uvec this_sort_index = arma::sort_index(this_contracting_indices);
            const arma::uvec that_sort_index = arma::sort_index(that_contracting_indices);
            const arma::uvec sorted_this_contracting_indices = this_contracting_indices(this_sort_index);
            const arma::uvec sorted_that_contracting_indices = that_contracting_indices(that_sort_index);

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

            auto result = BlockSparseTensor<std::common_type_t<T, U>>(new_dimension);

            for(arma::uword i=0; i<this->block_n_elem.n_elem; i++) {

                const arma::uvec A_begin_point = this->begin_points.col(i);
                const arma::uvec A_begin_point_in_contracting_dimension = A_begin_point.rows(this_contracting_indices);
                const arma::uvec A_end_point = this->end_points.col(i);

                const ContractionInfo contracting_info = block_in_range(contracting_indices, A_begin_point, A_end_point,
                                                                        tensor.begin_points, tensor.end_points);

                const arma::umat A_subblock_begin_points = contracting_info.A_begin_points;
                const arma::umat A_subblock_end_points = contracting_info.A_end_points;
                const arma::umat A_subblock_dimension =
                        A_subblock_end_points - A_subblock_begin_points + arma::ones<arma::umat>(arma::size(A_subblock_begin_points));

                const arma::umat B_subblock_begin_points = contracting_info.B_begin_points;
                const arma::umat B_subblock_end_points = contracting_info.B_end_points;
                const arma::umat B_subblock_dimension =
                        B_subblock_end_points - B_subblock_begin_points + arma::ones<arma::umat>(arma::size(B_subblock_begin_points));

                for(arma::uword j_block=0; j_block<contracting_info.block_indices.n_elem; j_block++) {
                    const arma::uword B_block_index = contracting_info.block_indices(j_block);

                    const arma::uvec B_begin_point = tensor.begin_points.col(B_block_index);


                    const arma::uvec B_begin_point_in_contracting_dimension = B_begin_point.rows(that_contracting_indices);

                    const arma::uvec j_contracting_start = contracting_info.contraction_begin_points.col(j_block);
                    const arma::uvec j_contracting_end = contracting_info.contraction_end_points.col(j_block);


                    const arma::uvec j_contracting_table = contracting_info.contraction_tables.col(j_block);
                    const arma::uvec j_contract_dim = j_contracting_end - j_contracting_start + arma::ones<arma::uvec>(j_contracting_start.n_elem);


                    if(result.rank > 0) {
                        const arma::uvec j_begin_point = contracting_info.new_begin_points.col(j_block);
                        const arma::uvec j_end_point = contracting_info.new_end_points.col(j_block);

                        const arma::uvec j_dim = j_end_point - j_begin_point + arma::ones<arma::uvec>(j_begin_point.n_elem);
                        const arma::uvec j_new_table = util::generate_index_table(j_dim);

                        const arma::uword this_block_offset =
                                arma::sum(
                                        (A_subblock_begin_points.col(j_block) - A_begin_point)
                                        % this->index_tables.col(i))
                                + this->block_offsets(i);

                        const arma::uword that_block_offset =
                                arma::sum((B_subblock_begin_points.col(j_block) - B_begin_point)
                                          % tensor.index_tables.col(B_block_index))
                                + tensor.block_offsets(B_block_index);

                        const arma::Mat<T> this_block_rep =
                                sort_into_matrix(this->data.get() + this_block_offset,
                                                 A_subblock_dimension.col(j_block),
                                                 this->index_tables.col(j_block),
                                                 this_contracting_indices);

                        const arma::Mat<T> that_block_rep =
                                sort_into_matrix(tensor.data.get() + that_block_offset,
                                                 B_subblock_dimension.col(j_block),
                                                 tensor.index_tables.col(j_block),
                                                 that_contracting_indices);

                        const arma::Mat<T> multiplication =
                                this_block_rep * that_block_rep.t();

                        const std::vector<T> vectorised_data =
                                arma::conv_to<std::vector<T>>::from(arma::vectorise(multiplication));

                        result.append_block(vectorised_data.data(),
                                            j_begin_point,
                                            j_end_point,
                                            j_new_table);

                    } else { // Full contraction, generating a scalar
                        for (arma::uword contract_index = 0; contract_index < arma::prod(j_contract_dim); contract_index++) {

                            const arma::uvec relative_contraction_indices = util::index_to_indices(contract_index, j_contract_dim);
                            const arma::uvec contraction_indices = relative_contraction_indices + j_contracting_start;

                            // assign the summation indices to the original tensor indices
                            const arma::uvec this_dimension_indices = contraction_indices(this_sort_index) - A_begin_point;
                            const arma::uvec that_dimension_indices = contraction_indices(that_sort_index) - B_begin_point;

                            const std::common_type_t<T, U>
                                    elem = this->data.get()[arma::sum(this->index_tables.col(i) % this_dimension_indices)
                                                            + this->block_offsets(i)]
                                           * tensor.data.get()[
                                                   arma::sum(tensor.index_tables.col(B_block_index) % that_dimension_indices)
                                                   + tensor.block_offsets(B_block_index)];

                            *(result.data.get()) += elem;
                        }
                    }
                }
            }

            return result;
        }

        /// Transposition of the tensors according to the permutation, without changing original data
        /// \param permutation the permutation indices
        inline
        void transpose(const arma::uvec & permutation) {

            if(permutation.n_elem != rank) {
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
        BlockSparseTensor<T> hard_transpose(const arma::uvec & permutation) const {

            if(permutation.n_elem != rank) {
                throw Error("The number of permutation does not match the rank of tensor");
            }

            const arma::uvec new_dimension = this->dimension(permutation);
            const arma::umat new_blocks_dimension = this->blocks_dimension.rows(permutation);
            const arma::umat new_begin_points = this->begin_points.rows(permutation);
            const arma::umat new_end_points = this->end_points.rows(permutation);

            auto result = BlockSparseTensor<T>(new_dimension);

            for(arma::uword i=0; i<new_begin_points.n_cols; i++) {
                const arma::uvec new_table = util::generate_index_table(new_blocks_dimension.col(i));
                const arma::uword n_elem = arma::prod(new_blocks_dimension.col(i));

                const arma::uvec old_table_sort_index = arma::sort_index(this->index_tables.col(i));

                T * block_data = (T *) malloc(n_elem * sizeof(T));
                for(arma::uword j=0; j<n_elem; j++) {
                    const arma::uvec old_indices = util::index_to_indices(j, this->index_tables.col(i), old_table_sort_index);
                    const arma::uvec new_indices = old_indices(permutation);

                    block_data[arma::sum(new_indices % new_table)] =
                            this->data.get()[arma::sum(old_indices % this->index_tables.col(i)) + this->block_offsets(i)];
                }

                result.append_block(block_data, new_begin_points.col(i), new_end_points.col(i), new_table);
            }

            return result;

        }

    protected:
        /// Stores data
        std::unique_ptr<T> data = nullptr;
    };

}

#endif //TORQUE_GPU_BLOCK_SPARSE_CUH
