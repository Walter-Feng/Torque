#ifndef TORQUE_TENSOR_DENSE_H
#define TORQUE_TENSOR_DENSE_H

#include <armadillo>
#include <memory>

#include "error.h"
#include "util/space.h"

namespace torque {

template<typename T>
class DenseTensor
{
public:

    /// The rank of the tensor
    arma::uword rank;

    /// The dimensions at each index of the tensor. The first index is the leading dimension of the
    /// tensor with stride equal to 1, i.e. difference of 1 for the first index will result in
    /// neighboring address in the data.
    arma::uvec dimension;

    /// An intermediate table that helps generating the index for the flattened one-dimensional data
    arma::uvec index_table;

    inline
    explicit DenseTensor(const arma::uvec & dimension) {

        this->dimension = dimension;

        rank = dimension.n_elem;

        if(rank > 0) {
            this->index_table = util::generate_index_table(dimension);

            this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::prod(dimension)));
            memset(this->data.get(), 0, sizeof(T) * arma::prod(dimension));
        } else {
         this->data = std::unique_ptr<T>((T *) malloc(sizeof(T)));
            memset(this->data.get(), 0, sizeof(T));
        }
    }

    inline
    explicit DenseTensor(const T * source_data, const arma::uvec & dimension) {

        this->dimension = dimension;

        rank = dimension.n_elem;

        if(rank > 0) {
            this->index_table = util::generate_index_table(dimension);

            this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::prod(dimension)));

            if(source_data) {
                memcpy(this->data.get(), source_data, sizeof(T) * arma::prod(dimension));
            } else {
                throw Error("Source data not allocated!");
            }
        } else {
            this->data = std::unique_ptr<T>((T *) malloc(sizeof(T)));

            if(source_data) {
                memcpy(this->data.get(), source_data, sizeof(T));
            } else {
                throw Error("Source data not allocated!");
            }
        }

    }

    inline
    explicit DenseTensor(std::unique_ptr<T> && source_data, const arma::uvec & dimension) {

        if(!source_data) {
            throw Error("Source data not allocated!");
        }

        this->dimension = dimension;

        rank = dimension.n_elem;

        this->index_table = util::generate_index_table(dimension);

        this->data = std::move(source_data);
    }

    inline
    DenseTensor(const DenseTensor & tensor) {
        this->rank = tensor.rank;
        this->dimension = tensor.dimension;
        this->index_table = tensor.index_table;

        this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::prod(this->dimension)));

        if(tensor.data) {
            memcpy(this->data.get(), tensor.data.get(), sizeof(T) * arma::prod(dimension));
        } else {
            throw Error("Source data not allocated!");
        }
    }

    inline
    ~DenseTensor(){
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
            this->data.get()[arma::sum(indices % this->index_table)] = number;
        } else {
            throw Error("Tensor not initialized");
        }
    }

    /// get the number from tensor with given indices
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
            return this->data.get()[arma::sum(indices % this->index_table)];
        } else {
            throw Error("Tensor not initialized");
        }
    }

    /// Contraction with another tensor
    /// \param tensor another tensor to be contracted with
    /// \param contracting_indices the corresponding two indices for the dimensions to contract
    /// from two tensors. It should be a (n x 2) matrix, with first col representing "this" tensor.
    template<typename U>
    DenseTensor<std::common_type_t<T, U>>
    contract(const DenseTensor<U> & tensor, const arma::umat & contracting_indices) const {
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
        const arma::uvec contract_table = util::generate_index_table(contract_dimension);

        if(!arma::all(contract_dimension - tensor.dimension(that_contracting_indices) == 0)) {
            throw Error("The dimensions from two tensors to be contracted do not match");
        }

        // Prepare dimension for the new tensor
        arma::uvec this_dimension_copy = this->dimension;
        arma::uvec that_dimension_copy = tensor.dimension;
        this_dimension_copy.shed_rows(this_contracting_indices);
        that_dimension_copy.shed_rows(that_contracting_indices);

        const arma::uvec new_dimension = arma::join_vert(this_dimension_copy, that_dimension_copy);
        const arma::uvec new_dimension_table = util::generate_index_table(new_dimension);

        auto result = DenseTensor<std::common_type_t<T, U>>(new_dimension);

        if(result.rank > 0) {
            for(arma::uword i=0; i<arma::prod(new_dimension); i++) {

                const arma::uvec new_dimension_indices = util::index_to_indices(i, new_dimension_table);

                arma::uvec this_dimension_indices = new_dimension_indices.rows(0, this->rank - contract_dimension.n_elem - 1);
                arma::uvec that_dimension_indices =
                        this->rank - contract_dimension.n_elem <= result.rank - 1 ?
                        new_dimension_indices.rows(this->rank - contract_dimension.n_elem, result.rank - 1) :
                        arma::uvec{};

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

                    const std::common_type_t<T, U>
                            elem = this->query(this_dimension_indices) * tensor.query(that_dimension_indices);

                    result.data.get()[i] += elem;
                }
            }
        } else { // Full contraction, generating a scalar
            for(arma::uword j=0; j<arma::prod(contract_dimension); j++){

                const arma::uvec contraction_indices = util::index_to_indices(j, contract_table);

                // assign the summation indices to the original tensor indices
                const arma::uvec this_dimension_indices = contraction_indices(this_sort_index);
                const arma::uvec that_dimension_indices =contraction_indices(that_sort_index);

                const std::common_type_t<T, U>
                        elem = this->query(this_dimension_indices) * tensor.query(that_dimension_indices);

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
        this->index_table = this->index_table(permutation);
        this->dimension = this->dimension(permutation);

    }


    /// Whether this tensor is sorted, i.e. has not been soft transposed.
    inline
    bool is_sorted() const {
        return this->index_table.is_sorted();
    }

    /// Whether this tensor has stride of the leading dimension equal to 1.
    inline
    bool has_leading_dimension() const {
        return this->index_table(1) == 0;
    }

    /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
    /// This helps keeping the stride of leading dimension equal to 1.
    /// \param permutation the permutation indices
    inline
    DenseTensor<T> hard_transpose(const arma::uvec & permutation) const {

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
        const arma::uvec sort_index = arma::sort_index(this->index_table);

        for(arma::uword i=0; i<total_elem; i++) {

            const arma::uvec new_indices = util::index_to_indices(i, this->index_table, sort_index);

            new_data_pointer[arma::sum(new_indices(permutation) % new_table)] = data_pointer[i];
        }

        return DenseTensor<T>(std::move(new_data), new_dimension);

    }

protected:
    /// Stores data
    std::unique_ptr<T> data = nullptr;
};

}

#endif //TORQUE_TENSOR_DENSE_H
