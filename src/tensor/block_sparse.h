#ifndef TORQUE_TENSOR_BLOCK_SPARSE_H
#define TORQUE_TENSOR_BLOCK_SPARSE_H

#include <armadillo>
#include <memory>

#include "error.h"
#include "util/space.h"

namespace torque {

struct
ContractionInfo {
    arma::uvec block_indices; // The selection of tensor blocks with non-trivial contribution
    arma::umat new_begin_points; // The begin points of new blocks from contraction
    arma::umat new_end_points; // The end points of new blocks from contraction

    arma::uvec contraction_A_begin_point; // the starting point of the block with non-trivial contribution to contraction from A block
    arma::umat contraction_B_begin_points; // the starting points of the blocks with non-trivial contribution to contraction from B blocks

    arma::uvec contraction_begin_point; // Starting point of the reduced subblock for contraction
    arma::umat contraction_end_points; // End point of the reduced subblock for contraction
    arma::umat contraction_tables; // index table for the contraction that helps iterating over the contracting elements
};

ContractionInfo
block_in_range(const arma::umat & contracting_indices,
               const arma::uvec & A_begin_point,
               const arma::uvec & A_end_point,
               const arma::umat & B_begin_points,
               const arma::umat & B_end_points) {

    // assert A block and B block have consistent number of subblocks and rank
    assert(A_begin_point.n_rows == A_end_point.n_rows);
    assert(B_begin_points.n_rows == B_end_points.n_rows);
    assert(B_begin_points.n_cols == B_end_points.n_cols);

    // The intervals of the non-trivial contribution from the blocks are min(end) - max(begin)
    arma::umat max_begin_indices_in_contracting_dimension(arma::size(contracting_indices.n_rows,
                                                                     B_begin_points.n_cols), arma::fill::zeros);

    arma::umat min_end_indices_in_contracting_dimension(arma::size(contracting_indices.n_rows,
                                                                   B_begin_points.n_cols), arma::fill::zeros);

    // Check whether it has non-trivial intervals for each block
    arma::Col<int> true_false_list(B_begin_points.n_cols, arma::fill::zeros);

    for(arma::uword i=0; i<B_begin_points.n_cols; i++) {
        //TODO: get the projection of these indices at contracting dimensions
        const arma::uvec max_begin_indices = arma::max(A_begin_point, B_begin_points.col(i));
        const arma::uvec min_end_indices = arma::min(A_end_point, B_end_points.col(i));

        if(arma::all(max_begin_indices < min_end_indices)) {
            true_false_list(i) = 1;

            max_begin_indices_in_contracting_dimension.col(i) = max_begin_indices;
            min_end_indices_in_contracting_dimension.col(i) = min_end_indices;
        }
    }

    const arma::uvec non_trivial_block_index = arma::find(true_false_list);

    if(non_trivial_block_index.n_elem) {
        arma::uvec new_begin_points_from_A = A_begin_point;
        new_begin_points_from_A.shed_rows(contracting_indices.col(0));
        arma::umat new_begin_points_from_B = B_begin_points;

    } else {
        return {arma::uvec{}, arma::umat{}, arma::umat{}};
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

        this->dimension = dimension;

        this->blocks_dimension = arma::zeros(dimension.n_elem, 1);
        this->block_n_elem = arma::prod(this->blocks_dimension);
        this->block_offsets = util::nest_sum(this->block_n_elem);

        rank = dimension.n_elem;

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
            this->blocks_dimension = end_points - begin_points + arma::ones<arma::uvec>(arma::size(begin_points));

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
        this->blocks_dimension = end_points - begin_points + arma::ones<arma::uvec>(arma::size(begin_points));

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

        const arma::uvec contract_dimension = this->dimension(this_contracting_indices);
        const arma::uvec contract_table = util::generate_index_table(contract_dimension);

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

        if(result.rank > 0) {
            for(arma::uword i=0; i<arma::prod(new_dimension); i++) {

                const arma::uvec new_dimension_indices = util::index_to_indices(i, new_dimension_table);

                arma::uvec this_dimension_indices = new_dimension_indices.rows(0, this->rank - contract_dimension.n_elem - 1);
                arma::uvec that_dimension_indices =
                        this->rank - contract_dimension.n_elem <= result.rank - 1 ?
                        new_dimension_indices.rows(this->rank - contract_dimension.n_elem, result.rank - 1) :
                        arma::uvec{};


                // It might be that the contracting indices may not be sorted, for example the matrix inner product
                // of two matrices (A, B) = sum( A % B^T ), where % is the element-wise multiplication
                // To restore the shed indices, we need the sorted contracting indices
                const arma::uvec sorted_this_contracting_indices = arma::sort(this_contracting_indices);
                const arma::uvec sorted_that_contracting_indices = arma::sort(that_contracting_indices);

                for(arma::uword k=0; k<sorted_this_contracting_indices.n_elem; k++){
                    this_dimension_indices.insert_rows(sorted_this_contracting_indices(k), 1);
                    that_dimension_indices.insert_rows(sorted_that_contracting_indices(k), 1);
                }

                for(arma::uword j=0; j<arma::prod(contract_dimension); j++){
                    const arma::uvec contraction_indices = util::index_to_indices(j, contract_table);

                    // assign the summation indices to the original tensor indices
                    for(arma::uword k=0; k<contraction_indices.n_elem; k++) {
                        this_dimension_indices(this_contracting_indices(k)) = contraction_indices(k);
                        that_dimension_indices(that_contracting_indices(k)) = contraction_indices(k);
                    }

                    assert(this_dimension_indices.n_elem == this->rank);
                    assert(that_dimension_indices.n_elem == tensor.rank);

                    const T elem = this->query(this_dimension_indices) * tensor.query(that_dimension_indices);

                    result.data.get()[i] += elem;
                }
            }
        } else { // Full contraction, generating a scalar
            for(arma::uword j=0; j<arma::prod(contract_dimension); j++){

                const arma::uvec contraction_indices = util::index_to_indices(j, contract_table);

                // assign the summation indices to the original tensor indices
                const arma::uvec this_dimension_indices = contraction_indices(this_contracting_indices);
                const arma::uvec that_dimension_indices = contraction_indices(that_contracting_indices);

                const T elem = this->query(this_dimension_indices) * tensor.query(that_dimension_indices);

                * (result.data.get()) += elem;
            }
        }

        return result;
    }

    /// Transposition of the tensors according to the permutation, without changing original data
    /// \param permutation the permutation indices
    inline
    void soft_transpose(const arma::uvec & permutation) {

        if(permutation.n_elem != rank) {
            throw Error("The number of permutation does not match the rank of tensor");
        }

        this->index_tables = this->index_tables.rows(permutation);
        this->dimension = this->dimension(permutation);
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

        arma::uvec new_table(rank);

        arma::uword table_index = 1;

        for (arma::uword i = 0; i < rank; i++) {
            new_table(i) = table_index;
            table_index *= new_dimension(i);
        }

        const arma::uword total_elem = arma::prod(this->dimension);

        std::unique_ptr<T> new_data((T *) malloc(sizeof(T) * total_elem));

        const auto data_pointer = this->data.get();
        const auto new_data_pointer = new_data.get();

        // It is possible that this tensor has been soft transposed, i.e.
        // the index table may not be in sorted order.
        const arma::uvec sort_index = arma::sort_index(this->index_tables);

        for(arma::uword i=0; i<total_elem; i++) {

            const arma::uvec new_indices = util::index_to_indices(i, this->index_tables, sort_index);

            new_data_pointer[arma::sum(new_indices(permutation) % new_table)] = data_pointer[i];
        }

        return BlockSparseTensor<T>(std::move(new_data_pointer), new_dimension);

    }

protected:
    /// Stores data
    std::unique_ptr<T> data = nullptr;
};

}

#endif //TORQUE_TENSOR_BLOCK_SPARSE_H
