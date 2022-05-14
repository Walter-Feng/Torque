#ifndef TORQUE_DENSE_H
#define TORQUE_DENSE_H

#include <armadillo>
#include <memory>
#include "error.h"

namespace torque {

template<typename T>
class DenseTensor
{
public:

    inline
    explicit DenseTensor(const arma::uvec & dimension) {

        this->dimension = dimension;

        rank = dimension.n_elem;

        arma::uword table_index = 1;

        this->index_table = arma::uvec(rank);

        for (arma::uword i = 0; i < rank; i++) {
            this->index_table(i) = table_index;
            table_index *= dimension(i);
        }

        this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::prod(dimension)));
        memset(this->data.get(), 0, sizeof(T) * arma::prod(dimension));

    }

    inline
    explicit DenseTensor(const T * source_data, const arma::uvec & dimension) {

        this->dimension = dimension;

        rank = dimension.n_elem;

        arma::uword table_index = 1;

        this->index_table = arma::uvec(rank);

        for (arma::uword i = 0; i < rank; i++) {
            this->index_table(i) = table_index;
            table_index *= dimension(i);
        }

        this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * arma::prod(dimension)));

        if(source_data) {
            memcpy(this->data.get(), source_data, sizeof(T) * arma::prod(dimension));
        } else {
            throw Error("Source data not allocated!");
        }
    }

    inline
    explicit DenseTensor(std::unique_ptr<T> && source_data, const arma::uvec & dimension) {

        this->dimension = dimension;

        rank = dimension.n_elem;

        arma::uword table_index = 1;

        this->index_table = arma::uvec(rank);

        for (arma::uword i = 0; i < rank; i++) {
            this->index_table(i) = table_index;
            table_index *= dimension(i);
        }

        this->data = source_data;
    }

    inline
    DenseTensor(const DenseTensor & tensor) {
        this->rank = tensor.rank;
        this->dimension = tensor.dimension;
        this->index_table = tensor.index_table;
    }

    inline
    ~DenseTensor(){
        this->data.release();
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
        this->data.get()[arma::sum(indices % this->index_table)] = number;
    }

    /// get the number from tensor with given indces
    /// \param indices indices for each dimension
    inline
    T query(const arma::uvec & indices) const {
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

        const arma::uvec contract_dimension = this->dimension(this_contracting_indices);

        if(!arma::all(contract_dimension - tensor.dimension(this_contracting_indices) == 0)) {
            throw Error("The dimensions from two tensors to be contracted do not match");
        }




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

        arma::uvec indices(this->rank);

        const auto data_pointer = this->data.get();
        const auto new_data_pointer = new_data.get();

        // It is possible that this tensor has been soft transposed, i.e.
        // the index table may not be in sorted order.
        const arma::uvec sort_index = arma::sort_index(this->index_table);

        for(arma::uword i=0; i<total_elem; i++) {

            arma::uword temp_i = i;

            for(arma::uword j = this->rank - 1; j > 0; j--) {
                indices(sort_index(j)) = temp_i / this->index_table(sort_index(j));
                temp_i = temp_i % this->index_table(sort_index(j));
            }

            indices(sort_index(0)) = temp_i;

            new_data_pointer[arma::sum(indices(permutation) % new_table)] = data_pointer[i];
        }

        return DenseTensor<T>(std::move(new_data_pointer), new_dimension);

    }

protected:
    /// The rank of the tensor
    arma::uword rank;

    /// The dimensions at each index of the tensor. The first index is the leading dimension of the
    /// tensor with stride equal to 1, i.e. difference of 1 for the first index will result in
    /// neighboring address in the data.
    arma::uvec dimension;

    /// An intermediate table that helps generating the index for the flattened one-dimensional data
    arma::uvec index_table;

    /// Stores data
    std::unique_ptr<T> data = nullptr;
};

}

#endif //TORQUE_DENSE_H
