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

        this->data = (T *) malloc(sizeof(T) * arma::prod(dimension));

        if(source_data) {
            memcpy(this->data, source_data, sizeof(T) * arma::prod(dimension));
        } else {
            throw Error("Source data not allocated!");
        }
    }

    inline
    explicit DenseTensor(const DenseTensor & tensor) {
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
    inline
    DenseTensor<T> contract(const DenseTensor<T> & tensor, const arma::umat & contracting_indices) const {


    }

    /// Transposition of the tensors according to the permutation
    /// \param permutation the permutation indices
    inline
    void transpose(const arma::uvec & permutation) {
        this->index_table = this->index_table(permutation);
        this->dimension = this->dimension(permutation);
    }

protected:
    /// The rank of the tensor
    arma::uword rank;

    /// The dimensions at each index of the tensor. The first index is the leading dimension of the
    /// tensor, i.e. difference of 1 for the first index will result in neighboring address in the
    /// data.
    arma::uvec dimension;

    /// An intermediate table that helps generating the index for the flattened one-dimensional data
    arma::uvec index_table;

    /// Stores data;
    std::unique_ptr<T> data = nullptr;
};

}

#endif //TORQUE_DENSE_H
