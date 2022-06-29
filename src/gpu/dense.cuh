#ifndef TORQUE_GPU_DENSE_CUH
#define TORQUE_GPU_DENSE_CUH

#include <armadillo>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <cutt.h>

//
// Error checking wrapper for cutt
//
#define cuttCheck(stmt) do {                                 \
  cuttResult err = stmt;                            \
  if (err != CUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                              \
  const auto err = x;                                                                  \
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   \
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } \
}

#include "error.h"
#include "gpu/util/thrust_arma_fusion.cuh"
#include "util/space.h"


namespace torque {
namespace gpu {

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
        explicit DenseTensor(const arma::uvec &dimension) {

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
        explicit DenseTensor(const T *source_data, const arma::uvec &dimension) {

            this->dimension = dimension;

            rank = dimension.n_elem;

            if (rank > 0) {
                this->index_table = torque::util::generate_index_table(dimension);

                this->data = thrust::device_vector<T>(arma::prod(dimension));

                if (source_data) {
                    thrust::copy(source_data, source_data + arma::prod(dimension), this->data.begin());
                } else {
                    throw Error("Source data not allocated!");
                }
            } else {
                this->data = thrust::device_vector<T>(1);

                if (source_data) {
                    this->data[0] = * source_data;
                } else {
                    throw Error("Source data not allocated!");
                }
            }

        }

        inline
        explicit DenseTensor(thrust::device_vector<T> &&source_data, const arma::uvec &dimension) {

            if (!source_data) {
                throw Error("Source data not allocated!");
            }

            this->dimension = dimension;

            rank = dimension.n_elem;

            this->index_table = torque::util::generate_index_table(dimension);

            this->data = std::move(source_data);
        }

        inline
        DenseTensor(const DenseTensor &tensor) {
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

        ///
        inline
        T to_number() const {
            assert(this->rank == 0);
            return this->data[0];
        }

        /// Initialization of the data, with proper memory allocation and memory copy
        inline
        void initialize(const T *source_data) {
            if (this->data.empty()) {
                throw Error("data not allocated!");
            }
            if (source_data) {
                thrust::copy(source_data, source_data + arma::prod(dimension), this->data.begin());
            } else {
                throw Error("Source data not allocated!");
            }
        }

        /// Modify a number in the tensor
        /// \param indices indices for each dimension
        /// \param number the target number (to replace the original one)
        inline
        void modify(const arma::uvec &indices, const T number) {

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
        T query(const arma::uvec &indices) const {

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

        /// Contraction with another tensor
        /// \param tensor another tensor to be contracted with
        /// \param contracting_indices the corresponding two indices for the dimensions to contract
        /// from two tensors. It should be a (n x 2) matrix, with first col representing "this" tensor.
        DenseTensor<T>
        contract(const DenseTensor<T> &tensor, const arma::umat &contracting_indices) const {

            auto compute_type = cutensor_compute_type<T>();
            auto data_type = cutensor_data_type<T>();

            T one = 1;
            T zero = 0;

            const arma::uvec this_contracting_indices = contracting_indices.col(0);
            const arma::uvec that_contracting_indices = contracting_indices.col(1);

            const arma::uvec contract_dimension = this->dimension(this_contracting_indices);

            if(!arma::all(contract_dimension - tensor.dimension(that_contracting_indices) == 0)) {
                throw Error("The dimensions from two tensors to be contracted do not match");
            }

            arma::uvec this_dimension_copy = this->dimension;
            arma::uvec that_dimension_copy = tensor.dimension;
            this_dimension_copy.shed_rows(this_contracting_indices);
            that_dimension_copy.shed_rows(that_contracting_indices);

            const arma::uvec new_dimension = arma::join_vert(this_dimension_copy, that_dimension_copy);
            const arma::uvec new_dimension_table = torque::util::generate_index_table(new_dimension);

            auto result = thrust::device_vector<T>(arma::prod(new_dimension));

            cutensorHandle_t handle;
            cutensorInit(&handle);

            auto this_dim = std::vector<int64_t>(this->rank);
            auto this_table = std::vector<int64_t>(this->rank);

            for(arma::uword i=0; i<this->rank; i++) {
                this_dim[i] = this->dimension(i);
                this_table[i] = this->index_table(i);
            }

            auto that_dim = std::vector<int64_t>(tensor.rank);
            auto that_table = std::vector<int64_t>(tensor.rank);

            for(arma::uword i=0; i<tensor.rank; i++) {
                that_dim[i] = tensor.dimension(i);
                that_table[i] = tensor.index_table(i);
            }

            auto result_dim = std::vector<int64_t>(new_dimension.n_elem);
            auto result_table = std::vector<int64_t>(new_dimension.n_elem);
            for(arma::uword i=0; i<new_dimension.n_elem; i++) {
                result_dim[i] = new_dimension(i);
                result_table[i] = new_dimension_table(i);
            }

            cutensorTensorDescriptor_t this_descriptor;

            HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                                                      &this_descriptor,
                                                      this->rank,
                                                      this_dim.data(),
                                                      this_table.data(),
                                                      data_type,
                                                      CUTENSOR_OP_IDENTITY));

            cutensorTensorDescriptor_t that_descriptor;

            HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                                                      &that_descriptor,
                                                      tensor.rank,
                                                      that_dim.data(),
                                                      that_table.data(),
                                                      data_type,
                                                      CUTENSOR_OP_IDENTITY));

            cutensorTensorDescriptor_t result_descriptor;
            HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                                                      &result_descriptor,
                                                      new_dimension.n_elem,
                                                      result_dim.data(),
                                                      result_table.data(),
                                                      data_type,
                                                      CUTENSOR_OP_IDENTITY));

            printf("Initialized cuTENSOR and tensor descriptors\n");

            uint32_t this_alignmentRequirement;
            HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
                                                           thrust::raw_pointer_cast(this->data.data()),
                                                           &this_descriptor,
                                                           &this_alignmentRequirement) );

            uint32_t that_alignmentRequirement;
            HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
                                                           thrust::raw_pointer_cast(tensor.data.data()),
                                                           &that_descriptor,
                                                           &that_alignmentRequirement) );

            uint32_t result_alignmentRequirement;
            HANDLE_ERROR( cutensorGetAlignmentRequirement( &handle,
                                                           thrust::raw_pointer_cast(result.data()),
                                                           &result_descriptor,
                                                           &result_alignmentRequirement) );

            printf("Query best alignment requirement for our pointers\n");


            arma::ivec total(this->rank + tensor.rank);
            for(int i=0; i<this->rank + tensor.rank; i++) {
                total(i) = i;
            }
            std::vector<int> this_mode(this->rank);
            for(int i=0; i<this->rank; i++)

            cutensorContractionDescriptor_t desc;
            HANDLE_ERROR( cutensorInitContractionDescriptor( &handle,
                                                             &desc,
                                                             &this_descriptor, modeA.data(), alignmentRequirementA,
                                                             &descB, modeB.data(), alignmentRequirementB,
                                                             &descC, modeC.data(), alignmentRequirementC,
                                                             &descC, modeC.data(), alignmentRequirementC,
                                                             typeCompute) );
        }

        /// Transposition of the tensors according to the permutation, creating new object with new alignment of data.
        /// This helps keeping the stride of leading dimension equal to 1.
        /// \param permutation the permutation indices
        inline
        DenseTensor<T> hard_transpose(const arma::uvec &permutation) const {

            if (permutation.n_elem != rank) {
                throw Error("The number of permutation does not match the rank of tensor");
            }

            const arma::uvec new_dimension = this->dimension(permutation);

            const arma::uword total_elem = arma::prod(this->dimension);

            auto new_data = thrust::device_vector<T>(total_elem);

            auto dim_in_cutt = std::vector<int>(this->rank);
            auto permutation_in_cutt = std::vector<int>(this->rank);

            for(arma::uword i=0; i<this->rank; i++) {
                dim_in_cutt[i] = this->dimension(i);
                permutation_in_cutt[i] = permutation(i);
            }

            cuttHandle plan;
            cuttCheck(cuttPlan(&plan, this->rank, dim_in_cutt, permutation_in_cutt, sizeof(T), 0));

            cuttCheck(cuttExecute(plan,
                      (void *) static_cast<void *>(thrust::raw_pointer_cast(this->data.data())),
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
