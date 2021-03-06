#ifndef TORQUE_TENSOR_H
#define TORQUE_TENSOR_H

#include <armadillo>
#include <cutensor.h>

namespace torque {

template<typename T>
class Tensor {
public:

  inline
  Tensor(const arma::uvec & dimension) {

    this->dimension = dimension;

    const arma::uword dims = dimension.n_elem;

    arma::uword table_index = 1;

    this->index_table = arma::uvec(dims);

    for (arma::uword i = 0; i < dims; i++) {
      this->index_table(i) = table_index;
      table_index *= dimension(i);
    }

    this->rank = dims;

  }

  /// Transposition of the tensors according to the permutation
  /// \param permutation the permutation indices
  virtual void transpose(const arma::uvec & permutation) {}

protected:

  /// The rank of the tensor
  arma::uword rank;

  /// The dimensions at each index of the tensor. The first index is the leading dimension of the
  /// tensor, i.e. difference of 1 for the first index will result in neighboring address in the
  /// data.
  arma::uvec dimension;


};

}


#endif //TORQUE_TENSOR_H
