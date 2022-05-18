#ifndef TORQUE_TENSOR_SPARSE_H
#define TORQUE_TENSOR_SPARSE_H

#include <armadillo>
#include <memory>

#include "error.h"

#include "util/space.h"

namespace torque {

template<typename T>
class SparseTensor
{
public:

  /// The rank of the tensor
  arma::uword rank;

  /// The dimensions at each index of the tensor.
  arma::uvec dimension;

  /// The indices for each element.
  arma::uvec indices;

  /// An intermediate table that helps generating the index for the flattened one-dimensional data
  arma::uvec index_table;

  inline
  explicit SparseTensor(const arma::uvec & dimension) {

    this->dimension = dimension;

    rank = dimension.n_elem;

    this->data = std::unique_ptr<T>((T *) malloc(sizeof(T)));
    memset(this->data.get(), 0, sizeof(T));
  }

  inline
  explicit SparseTensor(const T * source_data,
                        const arma::uvec & indices,
                        const arma::uvec & dimension) {

    this->dimension = dimension;
    this->indices = indices;
    this->index_table = util::generate_index_table(dimension);

    rank = dimension.n_elem;

    if(rank > 0) {
      this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * indices.n_elem));

        if(!this->data) {
            throw Error("Memory initialization failed");
        }

      if(source_data) {
        memcpy(this->data.get(), source_data, sizeof(T) * indices.n_elem);
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
  explicit SparseTensor(std::unique_ptr<T> && source_data,
                        const arma::uvec & index_table,
                        const arma::uvec & dimension) {

    if(!source_data) {
      throw Error("Source data not allocated!");
    }

    this->dimension = dimension;

    rank = dimension.n_elem;

    this->index_table = index_table;

    this->data = source_data;
  }

  inline
  SparseTensor(const SparseTensor & tensor) {
    this->rank = tensor.rank;
    this->dimension = tensor.dimension;
    this->index_table = tensor.index_table;
    this->indices = tensor.indices;

    this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * tensor.indices.n_elem));

    if(!this->data) {
        throw Error("Memory initialization failed");
    }

    if(tensor.data) {
      memcpy(this->data.get(), tensor.data.get(), sizeof(T) * tensor.indices.n_elem);
    } else {
      throw Error("Source data not allocated!");
    }
  }

  inline
  ~SparseTensor(){
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
  void initialize(const T * source_data, const arma::uvec & indices) {

    this->data.release();

    this->data = std::unique_ptr<T>((T *) malloc(sizeof(T) * indices.n_elem));
    if(source_data) {
      this->indices = indices;
      memcpy(this->data.get(), source_data, sizeof(T) * indices.n_elem);
    } else {
      throw Error("Source data not allocated!");
    }
  }

  /// Modify a number in the tensor
  /// \param indices indices for each dimension
  /// \param number the target number (to replace the original one)
  inline
  void modify(const arma::uvec & indices, const T number) {

      if (indices.n_elem != this->rank) {
          throw Error("Rank does not match");
      }
      if (this->data) {
          const arma::uvec new_index = indices % this->index_table;
          arma::uvec found_index = arma::find(this->indices == new_index);
          if (found_index.n_elem) {
              assert(found_index.n_elem == 1);

              if (number == 0) {

                  // Because we are doing sparse tensor,
                  // we would like to remove this element from the search list.
                  this->data.get()[found_index(0)] =
                          this->data.get()[this->indices.n_elem - 1];

                  this->indices(found_index(0)) = this->indices(this->indices.n_elem - 1);

                  // Remove the tail, as we have copied the tail element to the
                  // shed element specified by indices
                  this->indices.shed_row(this->indices.n_elem - 1);
                  this->data.get() = realloc(this->data.get(), this->indices.n_elem * sizeof(T));
              } else {
                  this->data.get()[found_index(0)] = number;
              }
          } else {
              if (number == 0) {
                  // Nothing happens
              } else {
                  this->indices.insert_rows(this->indices.n_elem, 1);
                  this->indices(this->indices.n_elem - 1) = new_index;
                  this->data.get() = realloc(this->data.get(), this->indices.n_elem * sizeof(T));
              }
          }
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

    if(this->data) {
      const arma::uvec new_index = indices % this->index_table;
      arma::uvec found_index = arma::find(this->indices == new_index);
      if(found_index.n_elem) {
          assert(found_index.n_elem == 1);

          return this->data.get()[found_index(0)];
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
  SparseTensor<std::common_type_t<T, U>>
  contract(const SparseTensor<U> & tensor, const arma::umat & contracting_indices) const {
    const arma::uvec this_contracting_indices = contracting_indices.col(0);
    const arma::uvec that_contracting_indices = contracting_indices.col(1);

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

    const arma::uword result_rank = this->rank - contract_dimension.n_elem;

    if(result_rank > 0) {
        std::vector<std::common_type_t<T, U>> raw_result_data;
        std::vector<arma::uword> raw_result_indices;

      for(arma::uword i=0; i<indices.n_elem; i++) {
          const arma::uvec this_sort_index = arma::sort_index(this->index_table);
          const arma::uvec this_indices = util::index_to_indices(this->indices(i), this->index_table, this_sort_index);

        for(arma::uword j=0; j<tensor.indices.n_elem; j++){
            const arma::uvec that_sort_index = arma::sort_index(this->index_table);
          const arma::uvec that_indices = util::index_to_indices(tensor.indices(j), tensor.index_table, that_sort_index);

          if(arma::all(this_indices(this_contracting_indices) == that_indices(that_contracting_indices))) {
              raw_result_data.push_back(this->data.get()[i] * tensor.data.get()[j]);
          }

          // Generate the new indices (i.e. in new tensor)
          arma::uvec this_indices_copy = this_indices;
          arma::uvec that_indices_copy = that_indices;
          this_indices_copy.shed_rows(this_contracting_indices);
          that_indices_copy.shed_rows(that_contracting_indices);

          const arma::uvec new_indices = arma::join_vert(this_indices_copy, that_indices_copy);
          raw_result_indices.push_back(arma::sum(new_indices % new_dimension_table));
        }
      } // raw multiplication of elements finished

      // there might be elements with the same indices in new tensor which need to be merged

      const arma::uvec unique_indices = arma::unique(arma::uvec(raw_result_indices));

      std::unique_ptr<std::common_type_t<T, U>>
        refined_data((std::common_type_t<T, U> *) calloc(unique_indices.n_elem,
                                                         sizeof(std::common_type_t<T, U>)));
      for(size_t i=0; i<raw_result_indices.size(); i++) {
          const arma::uvec index = arma::find(unique_indices == raw_result_indices[i]);
          assert(index.n_elem == 1);
          refined_data.get()[index(0)] += raw_result_data[i];
      }

      return SparseTensor<std::common_type_t<T, U>>(std::move(refined_data), new_dimension_table, new_dimension);

    } else { // Full contraction, generating a scalar

        std::common_type_t<T,U> result = 0;

        for(arma::uword i=0; i<indices.n_elem; i++) {
            const arma::uvec this_indices = util::index_to_indices(this->indices(i), this->index_table);

            for(arma::uword j=0; j<tensor.indices.n_elem; j++){
                const arma::uvec that_indices = util::index_to_indices(tensor.indices(j), tensor.index_table);

                if(arma::all(this_indices(this_contracting_indices) == that_indices(that_contracting_indices))) {
                    result += this->data.get()[i] * tensor.data.get()[j];
                }
            }
        }

        return SparseTensor<std::common_type_t<T, U>>(&result, arma::uvec{}, arma::uvec{});

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

  /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
  /// This helps keeping the stride of leading dimension equal to 1.
  /// \param permutation the permutation indices
  inline
  SparseTensor<T> hard_transpose(const arma::uvec & permutation) const {

    if(permutation.n_elem != rank) {
      throw Error("The number of permutation does not match the rank of tensor");
    }

    const arma::uvec new_dimension = this->dimension(permutation);

    const arma::uvec new_table = util::generate_index_table(new_dimension);

    const arma::uword total_elem = this->indices.n_elem;

    std::unique_ptr<T> new_data((T *) malloc(sizeof(T) * total_elem));

    memcpy(new_data.get(), this->data.get(), sizeof(T) * total_elem);
    // It is possible that this tensor has been soft transposed, i.e.
    // the index table may not be in sorted order.
    const arma::uvec sort_index = arma::sort_index(this->index_table);
    arma::uvec new_indices(this->indices.n_elem);

    for(arma::uword i=0; i<total_elem; i++) {

      const arma::uvec old_indices = util::index_to_indices(i, this->index_table, sort_index);
      new_indices(i) = arma::sum(old_indices(permutation) % new_table);

    }

    return SparseTensor<T>(std::move(new_data), new_indices, new_dimension);

  }

protected:
  /// Stores data
  std::unique_ptr<T> data = nullptr;
};

}

#endif //TORQUE_TENSOR_SPARSE_H
