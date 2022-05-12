#ifndef TORQUE_TENSOR_H
#define TORQUE_TENSOR_H

#include <armadillo>
#include <cutensor.h>

namespace torque {
    template<typename T>
    class Tensor
    {
    public:
        Tensor(cutensorHandle_t cutensorhandle, const arma::uvec & dimension) {

            this->cutensor_handle = cutensorhandle;

            this->dimension = dimension;

            const arma::uword dims = dimension.n_elem;

            arma::uword table_index = 1;

            this->index_table = arma::uvec(dims);

            for (arma::uword i = 0; i < dims; i++) {
                this->index_table(i) = table_index;
                table_index *= dimension(i);
            }

        }

        virtual ~Tensor();

        /// Initialization of the data, with proper memory allocation
        virtual void initialize(const arma::uvec & dimension);
        virtual void initialize(const T * source_data, const arma::uvec & dimension);

        /// Transposition of the tensors according to the permutation
        /// \param permutation the permutation indices
        virtual void transpose(const arma::uvec & permutation);

        /// Contraction with another tensor
        /// \param tensor another tensor to be contracted with
        virtual Tensor<T> contract(const Tensor<T> & tensor) const;

        /// get the number from tensor with given indces
        /// \param indices indices for each dimension
        virtual T query(const arma::uvec & indices) const;

    protected:

        /// The rank of the tensor
        arma::uword rank;

        /// The dimensions at each index of the tensor. The first index is the leading dimension of the
        /// tensor, i.e. difference of 1 for the first index will result in neighboring address in the
        /// data.
        arma::uvec dimension;

        /// An intermediate table that helps generating the index for the flattened one-dimensional data
        arma::uvec index_table;
    private:

        /// Stores data;
        T * data = nullptr;

    };
}


#endif //TORQUE_TENSOR_H
