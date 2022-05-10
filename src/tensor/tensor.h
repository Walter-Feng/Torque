#ifndef TORQUE_TENSOR_H
#define TORQUE_TENSOR_H

#include <armadillo>
#include <cutensor.h>

template<typename T>
class Tensor
{
public:
    Tensor(cutensorHandle_t cutensorhandle);

    virtual ~Tensor();

    /// Transposition of the tensors according to the permutation
    /// \param permutation the permutation indices
    virtual void transpose(const arma::uvec & permutation);

    /// Contraction with another tensor
    Tensor<T> contract(const Tensor<T> & tensor);



protected:

    /// The rank of the tensor
    arma::uword rank;

    /// The dimensions at each index of the tensor
    arma::uvec dimension;


private:

    /// Stores the dimension of the data;
    arma::Col<T> data;

};

#endif //TORQUE_TENSOR_H
