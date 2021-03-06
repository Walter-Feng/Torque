#ifndef TORQUE_GPU_DENSE_CUH
#define TORQUE_GPU_DENSE_CUH

#define ARMA_ALLOW_FAKE_GCC

#include <optional>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutt.h>

#include "error.h"
#include "gpu/util/thrust_arma_fusion.cuh"
#include "gpu/util/lib_helper.cuh"
#include "util/space.h"

#ifdef USE_CUTENSOR

#include <cutensor.h>

#endif

namespace torque {
namespace gpu {

#ifdef USE_CUTENSOR

template<typename T>
cutensorComputeType_t cutensor_compute_type() {
  if constexpr(std::is_same<T, float>::value) {
    return CUTENSOR_COMPUTE_32F;
  } else if constexpr(std::is_same<T, double>::value) {
    return CUTENSOR_COMPUTE_64F;
  } else if constexpr(std::is_same<T, half>::value) {
    return CUTENSOR_COMPUTE_16F;
  }
}

template<typename T>
cudaDataType_t cutensor_data_type() {
  if constexpr(std::is_same<T, float>::value) {
    return CUDA_R_32F;
  } else if constexpr(std::is_same<T, double>::value) {
    return CUDA_R_64F;
  } else if constexpr(std::is_same<T, half>::value) {
    return CUDA_R_16F;
  }
}

#endif

template<typename T>
class DenseTensor {
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

    if (rank > 0) {
      this->index_table = torque::util::generate_index_table(dimension);

      this->data = thrust::device_vector<T>(arma::prod(dimension));

    } else {
      this->data = thrust::device_vector<T>(1);
      this->data[0] = 0;
    }
  }

  inline
  explicit DenseTensor(const T * source_data, const arma::uvec & dimension) {

    this->dimension = dimension;

    rank = dimension.n_elem;

    if (rank > 0) {
      this->index_table = torque::util::generate_index_table(dimension);

      this->data = thrust::device_vector<T>(arma::prod(dimension));

      if (source_data) {
        thrust::copy(source_data, source_data + arma::prod(dimension),
                     this->data.begin());
      } else {
        throw Error("Source data not allocated!");
      }
    } else {
      this->data = thrust::device_vector<T>(1);

      if (source_data) {
        this->data[0] = *source_data;
      } else {
        throw Error("Source data not allocated!");
      }
    }

  }

  inline
  explicit DenseTensor(thrust::device_vector<T> && source_data,
                       const arma::uvec & dimension) {

    if (source_data.empty()) {
      throw Error("Source data not allocated!");
    }

    this->dimension = dimension;

    rank = dimension.n_elem;

    this->index_table = torque::util::generate_index_table(dimension);

    this->data = std::move(source_data);
  }

  inline
  DenseTensor(const DenseTensor & tensor) {
    this->rank = tensor.rank;
    this->dimension = tensor.dimension;
    this->index_table = tensor.index_table;

    this->data = thrust::device_vector<T>(arma::prod(dimension));

    if (tensor.data) {
      thrust::copy(tensor.data.begin(), tensor.data.end(), this->data.begin());
    } else {
      throw Error("Source data not allocated!");
    }
  }

  inline
  DenseTensor(DenseTensor && tensor) noexcept:
      rank(std::move(tensor.rank)),
      dimension(std::move(tensor.dimension)),
      index_table(std::move(tensor.index_table)),
      data(std::move(tensor.data)) {}

  inline
  DenseTensor & operator=(DenseTensor<T> && other) noexcept {
    rank = std::move(other.rank);
    dimension = std::move(other.dimension);
    index_table = std::move(other.index_table);
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
  void initialize(const T * source_data) {
    if (this->data.empty()) {
      throw Error("data not allocated!");
    }
    if (source_data) {
      thrust::copy(source_data, source_data + arma::prod(dimension),
                   this->data.begin());
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

    if (arma::any(indices >= this->dimension)) {
      throw Error("Indices out of boundary");
    }

    if (!this->data.empty()) {
      this->data[arma::sum(indices % this->index_table)] = number;
    } else {
      throw Error("Tensor not initialized");
    }
  }

  /// get the number from tensor with given indices
  /// \param indices indices for each dimension
  inline
  T query(const arma::uvec & indices) const {

    if (indices.n_elem != this->rank) {
      throw Error("Rank does not match");
    }

    if (arma::any(indices >= this->dimension)) {
      throw Error("Indices out of boundary");
    }

    if (!this->data.empty()) {
      return this->data[arma::sum(indices % this->index_table)];
    } else {
      throw Error("Tensor not initialized");
    }
  }

#ifdef USE_CUTENSOR

  /// Contraction with another tensor
  /// \param tensor another tensor to be contracted with
  /// \param contracting_indices the corresponding two indices for the dimensions to contract
  /// from two tensors. It should be a (n x 2) matrix, with first col representing "this" tensor.
  DenseTensor<T>
  contract(cutensorHandle_t * cutensor_handle,
           const DenseTensor<T> & tensor,
           const arma::umat & contracting_indices) const {

    T one = 1;
    T zero = 0;

    const arma::uvec this_contracting_indices = contracting_indices.col(0);
    const arma::uvec that_contracting_indices = contracting_indices.col(1);

    const arma::uvec contract_dimension = this->dimension(
        this_contracting_indices);

    if (!arma::all(
        contract_dimension - tensor.dimension(that_contracting_indices) == 0)) {
      throw Error(
          "The dimensions from two tensors to be contracted do not match");
    }

    arma::uvec this_dimension_copy = this->dimension;
    arma::uvec that_dimension_copy = tensor.dimension;
    this_dimension_copy.shed_rows(this_contracting_indices);
    that_dimension_copy.shed_rows(that_contracting_indices);

    const arma::uvec new_dimension = arma::join_vert(this_dimension_copy,
                                                     that_dimension_copy);
    const arma::uvec new_dimension_table = torque::util::generate_index_table(
        new_dimension);

    auto result = thrust::device_vector<T>(arma::prod(new_dimension));

    auto this_dim = std::vector<int64_t>(this->rank);
    auto this_table = std::vector<int64_t>(this->rank);

    for (arma::uword i = 0; i < this->rank; i++) {
      this_dim[i] = this->dimension(i);
      this_table[i] = this->index_table(i);
    }

    auto that_dim = std::vector<int64_t>(tensor.rank);
    auto that_table = std::vector<int64_t>(tensor.rank);

    for (arma::uword i = 0; i < tensor.rank; i++) {
      that_dim[i] = tensor.dimension(i);
      that_table[i] = tensor.index_table(i);
    }

    auto result_dim = std::vector<int64_t>(new_dimension.n_elem);
    auto result_table = std::vector<int64_t>(new_dimension.n_elem);
    for (arma::uword i = 0; i < new_dimension.n_elem; i++) {
      result_dim[i] = new_dimension(i);
      result_table[i] = new_dimension_table(i);
    }

    auto compute_type = cutensor_compute_type<T>();
    auto data_type = cutensor_data_type<T>();

    cutensorTensorDescriptor_t this_descriptor;

    HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle,
                                              &this_descriptor,
                                              this->rank,
                                              this_dim.data(),
                                              this_table.data(),
                                              data_type,
                                              CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t that_descriptor;

    HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle,
                                              &that_descriptor,
                                              tensor.rank,
                                              that_dim.data(),
                                              that_table.data(),
                                              data_type,
                                              CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t result_descriptor;
    HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle,
                                              &result_descriptor,
                                              new_dimension.n_elem,
                                              result_dim.data(),
                                              result_table.data(),
                                              data_type,
                                              CUTENSOR_OP_IDENTITY));

    uint32_t this_alignmentRequirement;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle,
                                                 thrust::raw_pointer_cast(
                                                     this->data.data()),
                                                 &this_descriptor,
                                                 &this_alignmentRequirement));

    uint32_t that_alignmentRequirement;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle,
                                                 thrust::raw_pointer_cast(
                                                     tensor.data.data()),
                                                 &that_descriptor,
                                                 &that_alignmentRequirement));

    uint32_t result_alignmentRequirement;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle,
                                                 thrust::raw_pointer_cast(
                                                     result.data()),
                                                 &result_descriptor,
                                                 &result_alignmentRequirement));

    std::vector<int> total(this->rank + tensor.rank);
    for (int i = 0; i < this->rank + tensor.rank; i++) {
      total[i] = i + 1;
    }
    std::vector<int> this_mode(this->rank);
    std::vector<int> that_mode(tensor.rank);

    memcpy(this_mode.data(), total.data(), sizeof(int) * this->rank);
    memcpy(that_mode.data(), total.data() + this->rank,
           sizeof(int) * tensor.rank);

    for (int i = 0; i < this_contracting_indices.n_elem; i++) {
      this_mode[this_contracting_indices(i)] = -(i + 1);
      that_mode[that_contracting_indices(i)] = -(i + 1);
      total[this_contracting_indices(i)] = 0;
      total[this->rank + that_contracting_indices(i)] = 0;
    }

    const auto result_mode =
        arma::conv_to<std::vector<int>>::from(
            arma::nonzeros(arma::Col<int>(total)));

    cutensorContractionDescriptor_t desc;

    HANDLE_ERROR(cutensorInitContractionDescriptor(cutensor_handle,
                                                   &desc,
                                                   &this_descriptor,
                                                   this_mode.data(),
                                                   this_alignmentRequirement,
                                                   &that_descriptor,
                                                   that_mode.data(),
                                                   that_alignmentRequirement,
                                                   &result_descriptor,
                                                   result_mode.data(),
                                                   result_alignmentRequirement,
                                                   &result_descriptor,
                                                   result_mode.data(),
                                                   result_alignmentRequirement,
                                                   compute_type));


    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind(
        cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));

    // Query workspace
    size_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspace(cutensor_handle,
                                                 &desc,
                                                 &find,
                                                 CUTENSOR_WORKSPACE_RECOMMENDED,
                                                 &worksize));

    // Allocate workspace
    void * work = nullptr;
    if (worksize > 0) {
      if (cudaSuccess != cudaMalloc(&work, worksize)) // This is optional!
      {
        work = nullptr;
        worksize = 0;
      }
    }

    // Create Contraction Plan
    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(cutensor_handle,
                                             &plan,
                                             &desc,
                                             &find,
                                             worksize));

    cutensorStatus_t err;

    // Execute the tensor contraction
    err = cutensorContraction(cutensor_handle,
                              &plan,
                              (void *) &one,
                              thrust::raw_pointer_cast(this->data.data()),
                              thrust::raw_pointer_cast(tensor.data.data()),
                              (void *) &zero,
                              thrust::raw_pointer_cast(result.data()),
                              thrust::raw_pointer_cast(result.data()),
                              work, worksize, 0 /* stream */);
    cudaDeviceSynchronize();

    // Check for errors
    if (err != CUTENSOR_STATUS_SUCCESS) {
      printf("ERROR: %s\n", cutensorGetErrorString(err));
    }

    return DenseTensor<T>(std::move(result), new_dimension);

  }

#else

  /// Contraction with another tensor
  /// \param tensor another tensor to be contracted with
  /// \param contracting_indices the corresponding two indices for the dimensions to contract
  /// from two tensors. It should be a (n x 2) matrix, with first col representing "this" tensor.
  DenseTensor<T>
  contract(cublasHandle_t cublas_handle,
           const DenseTensor<T> & tensor,
           const arma::umat & contracting_indices) const {

    T one = 1;
    T zero = 0;

    const arma::uvec this_contracting_indices = contracting_indices.col(0);
    const arma::uvec that_contracting_indices = contracting_indices.col(1);

    const arma::uvec contract_dimension = this->dimension(
        this_contracting_indices);

    if (!arma::all(
        contract_dimension - tensor.dimension(that_contracting_indices) == 0)) {
      throw Error(
          "The dimensions from two tensors to be contracted do not match");
    }

    arma::uvec this_dimension_copy = this->dimension;
    arma::uvec that_dimension_copy = tensor.dimension;
    this_dimension_copy.shed_rows(this_contracting_indices);
    that_dimension_copy.shed_rows(that_contracting_indices);

    const arma::uvec new_dimension = arma::join_vert(this_dimension_copy,
                                                     that_dimension_copy);
    const arma::uvec new_dimension_table = torque::util::generate_index_table(
        new_dimension);

    auto result = thrust::device_vector<T>(arma::prod(new_dimension));

    auto this_dim = std::vector<int64_t>(this->rank);
    auto this_table = std::vector<int64_t>(this->rank);

    for (arma::uword i = 0; i < this->rank; i++) {
      this_dim[i] = this->dimension(i);
      this_table[i] = this->index_table(i);
    }

    auto that_dim = std::vector<int64_t>(tensor.rank);
    auto that_table = std::vector<int64_t>(tensor.rank);

    for (arma::uword i = 0; i < tensor.rank; i++) {
      that_dim[i] = tensor.dimension(i);
      that_table[i] = tensor.index_table(i);
    }

    auto result_dim = std::vector<int64_t>(new_dimension.n_elem);
    auto result_table = std::vector<int64_t>(new_dimension.n_elem);
    for (arma::uword i = 0; i < new_dimension.n_elem; i++) {
      result_dim[i] = new_dimension(i);
      result_table[i] = new_dimension_table(i);
    }

    const auto permutation_generator =
            [](const arma::uvec & contracting_indices, const arma::uword target_rank) -> arma::uvec {

        arma::uvec transposition(target_rank);

        for(int i=0; i<target_rank; i++) {
            transposition(i) = i;
        }

        transposition.shed_rows(contracting_indices);

        return arma::join_vert(transposition, contracting_indices);
    };

    const arma::uvec this_permutation = permutation_generator(this_contracting_indices, this->rank);
    const arma::uvec that_permutation = permutation_generator(that_contracting_indices, tensor.rank);

    const std::optional<DenseTensor<T>> this_transposed =
            this_permutation.is_sorted() ? std::nullopt : std::optional(this->hard_transpose(this_permutation));
    const std::optional<DenseTensor<T>> that_transposed =
            that_permutation.is_sorted() ? std::nullopt : std::optional(tensor.hard_transpose(that_permutation));

    const arma::uword contracting_n_elem = arma::prod(contract_dimension);

    const arma::uword this_leading_dim = arma::prod(this->dimension) / contracting_n_elem;
    const arma::uword that_leading_dim = arma::prod(tensor.dimension) / contracting_n_elem;

    const T * this_pointer = this_transposed.has_value() ?
                             thrust::raw_pointer_cast(this_transposed.value().data.data()) :
                             thrust::raw_pointer_cast(this->data.data()) ;

    const T * that_pointer = that_transposed.has_value() ?
                             thrust::raw_pointer_cast(that_transposed.value().data.data()) :
                             thrust::raw_pointer_cast(tensor.data.data()) ;

    T * out_pointer = thrust::raw_pointer_cast(result.data());

    if constexpr(std::is_same<T, float>::value) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    this_leading_dim, that_leading_dim, contracting_n_elem, &one,
                    this_pointer, this_leading_dim,
                    that_pointer, that_leading_dim,
                    &zero, out_pointer, this_leading_dim);
    }

    if constexpr(std::is_same<T, double>::value) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    this_leading_dim, that_leading_dim, contracting_n_elem, &one,
                    this_pointer, this_leading_dim,
                    that_pointer, that_leading_dim,
                    &zero, out_pointer, this_leading_dim);
    }

    if constexpr(std::is_same<T, half>::value) {
        cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    this_leading_dim, that_leading_dim, contracting_n_elem, &one,
                    this_pointer, this_leading_dim,
                    that_pointer, that_leading_dim,
                    &zero, out_pointer, this_leading_dim);
    }
    return DenseTensor<T>(std::move(result), new_dimension);
  }
#endif

  /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
  /// This helps keeping the stride of leading dimension equal to 1.
  /// \param permutation the permutation indices
  inline
  DenseTensor<T> hard_transpose(const arma::uvec & permutation) const {

    if (permutation.n_elem != rank) {
      throw Error(
          "The number of permutation does not match the rank of tensor");
    }

    const arma::uvec new_dimension = this->dimension(permutation);

    const arma::uword total_elem = arma::prod(this->dimension);

    auto new_data = thrust::device_vector<T>(total_elem);

    auto dim_in_cutt = std::vector<int>(this->rank);
    auto permutation_in_cutt = std::vector<int>(permutation.n_elem);

    for (arma::uword i = 0; i < this->rank; i++) {
      dim_in_cutt[i] = this->dimension(i);
      permutation_in_cutt[i] = permutation(i);
    }

    cuttHandle plan;

    cuttCheck(cuttPlan(&plan, this->rank,
                       dim_in_cutt.data(),
                       permutation_in_cutt.data(),
                       sizeof(T), 0));

    cuttCheck(cuttExecute(plan,
                          (void *) const_cast<T *>(thrust::raw_pointer_cast(
                              this->data.data())),
                          (void *) thrust::raw_pointer_cast(new_data.data())));

    cuttCheck(cuttDestroy(plan));

    return DenseTensor<T>(std::move(new_data), new_dimension);

  }

protected:
  /// Stores data
  thrust::device_vector<T> data;
};


}
}

#endif //TORQUE_GPU_DENSE_CUH
