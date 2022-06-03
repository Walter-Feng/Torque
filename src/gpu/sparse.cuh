#ifndef TORQUE_GPU_SPARSE_CUH
#define TORQUE_GPU_SPARSE_CUH
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "gpu/util/thrust_arma_fusion.cuh"
#include "util/space.h"

namespace torque {
namespace gpu {
namespace sparse {
    thrust::device_vector<int32_t>  handle_indices(
            const thrust::device_vector<int32_t> & A_indices,
            const thrust::device_vector<int32_t> & B_indices,
            const thrust::device_vector<int32_t> & A_index_table,
            const thrust::device_vector<int32_t> & A_sorted_index,
            const thrust::device_vector<int32_t> & B_index_table,
            const thrust::device_vector<int32_t> & B_sorted_index,
            const thrust::device_vector<int32_t> & A_contracting_indices,
            const thrust::device_vector<int32_t> & B_contracting_indices,
            const thrust::device_vector<int32_t> & A_free_indices,
            const thrust::device_vector<int32_t> & B_free_indices,
            const thrust::device_vector<int32_t> & index_table_out
    );

    inline
    thrust::device_vector<int32_t>
    handle_indices(
            const thrust::device_vector<int32_t> & A_indices,
            const thrust::device_vector<int32_t> & B_indices,
            const arma::uvec & A_dimension,
            const arma::uvec & B_dimension,
            const arma::uvec & A_index_table,
            const arma::uvec & B_index_table,
            const arma::uvec & A_contracting_indices,
            const arma::uvec & B_contracting_indices,
            const arma::uvec & out_index_table
    ) {
        const arma::uvec A_sorted_index = arma::sort_index(A_index_table);
        const arma::uvec B_sorted_index = arma::sort_index(B_index_table);

        arma::uvec A_free_indices(A_dimension.n_elem);
        for(arma::uword i=0; i<A_free_indices.n_elem; i++) {
            A_free_indices(i) = i;
        }

        arma::uvec B_free_indices(B_dimension.n_elem);
        for(arma::uword i=0; i<B_free_indices.n_elem; i++) {
            B_free_indices(i) = i;
        }

        A_free_indices.shed_rows(A_contracting_indices);
        B_free_indices.shed_rows(B_contracting_indices);


        return handle_indices(
                A_indices,
                B_indices,
                gpu::util::arma_to_thrust_device(A_index_table),
                gpu::util::arma_to_thrust_device(A_sorted_index),
                gpu::util::arma_to_thrust_device(B_index_table),
                gpu::util::arma_to_thrust_device(B_sorted_index),
                gpu::util::arma_to_thrust_device(A_contracting_indices),
                gpu::util::arma_to_thrust_device(B_contracting_indices),
                gpu::util::arma_to_thrust_device(A_free_indices),
                gpu::util::arma_to_thrust_device(B_free_indices),
                gpu::util::arma_to_thrust_device(out_index_table)
        );
    }

}

template<typename T>
class SparseTensor
{
public:

    /// The rank of the tensor
    arma::uword rank;

    /// The dimensions at each index of the tensor.
    arma::uvec dimension;

    /// The indices for each element.
    thrust::device_vector<int32_t> indices;

    /// An intermediate table that helps generating the index for the flattened one-dimensional data
    arma::uvec index_table;

    inline
    explicit SparseTensor(const arma::uvec & dimension) {

        this->dimension = dimension;
        this->index_table = torque::util::generate_index_table(dimension);

        rank = dimension.n_elem;

        this->data = thrust::device_vector<T>(1);
        this->data[0] = 0;
    }

    inline
    explicit SparseTensor(const T * source_data,
                          const arma::uvec & indices,
                          const arma::uvec & index_table,
                          const arma::uvec & dimension) {

        this->dimension = dimension;
        this->indices = thrust::device_vector<int32_t>(arma::conv_to<std::vector<T>>::from(indices));
        this->index_table = index_table;

        rank = dimension.n_elem;

        if(rank > 0) {
            this->data = thrust::device_vector<T>(indices.n_elem);

            if(this->data.empty()) {
                throw Error("Memory initialization failed");
            }

            if(source_data) {
                cudaMemcpy(thrust::raw_pointer_cast(
                        this->data.data()),
                           source_data,
                           indices.n_elem * sizeof(T),
                           cudaMemcpyHostToDevice);

            } else {
                throw Error("Source data not allocated!");
            }
        } else {
            this->data = thrust::device_vector<T>(1);

            if(source_data) {
                cudaMemcpy(thrust::raw_pointer_cast(this->data.data()),source_data, sizeof(T),
                           cudaMemcpyHostToDevice);
            } else {
                throw Error("Source data not allocated!");
            }
        }

    }

    inline
    explicit SparseTensor(thrust::device_vector<T> && source_data,
                          const thrust::device_vector<int32_t> & indices,
                          const arma::uvec & index_table,
                          const arma::uvec & dimension) {

        if(source_data.empty()) {
            throw Error("Source data not allocated!");
        }

        this->dimension = dimension;

        rank = dimension.n_elem;

        this->indices = indices;

        this->index_table = index_table;

        this->data = std::move(source_data);
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

        this->data = thrust::device_vector<T>(indices.n_elem);
        if(source_data) {
            this->indices = thrust::device_vector<int32_t>(arma::conv_to<std::vector<int32_t>>::from(indices));
            thrust::copy(source_data, source_data + indices.n_elem, this->data.begin());
        } else {
            throw Error("Source data not allocated!");
        }
    }

    /// Modify a number in the tensor
    /// \param indices indices for each dimension
    /// \param number the target number (to replace the original one)
    inline
    void modify(const arma::uvec & modify_indices, const T number) {

        if (modify_indices.n_elem != this->rank) {
            throw Error("Rank does not match");
        }
        if (!this->data.empty()) {
            if(this->rank == 0) {
                this->data[0] = number;
            } else {
                const arma::uword new_index = arma::sum(modify_indices % this->index_table);
                arma::uvec found_index = util::thrust_find(this->indices, new_index);
                if (found_index.n_elem) {
                    assert(found_index.n_elem == 1);

                    if (number == 0) {

                        // Because we are doing sparse tensor,
                        // we would like to remove this element from the search list.
                        this->data[found_index(0)] =
                                this->data[this->indices.size() - 1];

                        this->indices[found_index(0)] = this->indices[this->indices.size() - 1];

                        // Remove the tail, as we have copied the tail element to the
                        // shed element specified by indices
                        this->indices.erase(this->indices.end() - 1);
                    } else {
                        this->data[found_index(0)] = number;
                    }
                } else {
                    if (number == 0) {
                        // Nothing happens
                    } else {
                        this->indices.insert(this->indices.end(), new_index);
                        this->data.push_back(number);
                    }
                }
            }

        } else {
            throw Error("Tensor not initialized");
        }
    }

    /// get the number from tensor with given indices
    /// \param query_indices indices for each dimension
    inline
    T query(const arma::uvec & query_indices) const {

        if(query_indices.n_elem != this->rank) {
            throw Error("Rank does not match");
        }

        if(!this->data.empty()) {
            if(this->rank == 0) {
                return this->data[0];
            } else {
                const arma::uword new_index = arma::sum(query_indices % this->index_table);
                arma::uvec found_index = util::thrust_find(this->indices, new_index);
                if(found_index.n_elem) {
                    assert(found_index.n_elem == 1);

                    return this->data[found_index(0)];
                } else {
                    return 0;
                }
            }
        } else {
            throw Error("Tensor not initialized");
        }
    }

    /// Contraction with another tensor
    /// \param tensor another tensor to be contracted with
    /// \param contracting_indices the corresponding two indices for the dimensions to contract
    /// from two tensors. It should be a (n x 2) matrix, with first col representing "this" tensor.
    thrust::device_vector<T>
    contract(cublasHandle_t handle, const SparseTensor<T> & tensor, const arma::umat & contracting_indices) const {

        const arma::uvec this_contracting_indices = contracting_indices.col(0);
        const arma::uvec that_contracting_indices = contracting_indices.col(1);

        const arma::uvec contract_dimension = this->dimension(this_contracting_indices);
        const arma::uvec contract_table = torque::util::generate_index_table(contract_dimension);

        if(!arma::all(contract_dimension - tensor.dimension(that_contracting_indices) == 0)) {
            throw Error("The dimensions from two tensors to be contracted do not match");
        }

        // Prepare dimension for the new tensor
        arma::uvec this_dimension_copy = this->dimension;
        arma::uvec that_dimension_copy = tensor.dimension;
        this_dimension_copy.shed_rows(this_contracting_indices);
        that_dimension_copy.shed_rows(that_contracting_indices);
        const arma::uvec new_dimension = arma::join_vert(this_dimension_copy, that_dimension_copy);
        const arma::uvec new_dimension_table = torque::util::generate_index_table(new_dimension);

        const auto new_indices_raw = sparse::handle_indices(this->indices, tensor.indices,
                                                            this->dimension, tensor.dimension,
                                                            this->index_table, tensor.index_table,
                                                            this_contracting_indices, that_contracting_indices,
                                                            new_dimension_table);

        thrust::device_vector<T> raw_output =
                thrust::device_vector<T>(this->indices.size() * tensor.indices.size());

        static_assert(
                std::is_same<T, float>::value
                || std::is_same<T, double>::value
                || std::is_same<T, half>::value,
                "GPU-enabled sparse tensor contraction can only support float, double and half");

        T one = 1;
        T zero = 0;

        const T * this_pointer = thrust::raw_pointer_cast(this->data.data());
        const T * that_pointer = thrust::raw_pointer_cast(tensor.data.data());
        T * out_pointer = thrust::raw_pointer_cast(raw_output.data());


        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, this->indices.size(), tensor.indices.size(), 1, &one,
                    this_pointer, this->indices.size(),
                    that_pointer, tensor.indices.size(),
                    &zero, out_pointer, this->indices.size());
//
//        if constexpr(std::is_same<T, float>::value) {
//            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, this->indices.size(), tensor.indices.size(), 1, &one,
//                        this_pointer, this->indices.size(),
//                        that_pointer, tensor.indices.size(),
//                        &zero, out_pointer, this->indices.size());
//        }
//
//        if constexpr(std::is_same<T, double>::value) {
//            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, this->indices.size(), tensor.indices.size(), 1, &one,
//                        this_pointer, 1,
//                        that_pointer, 1,
//                        &zero, out_pointer, 1);
//        }
//
//        if constexpr(std::is_same<T, half>::value) {
//            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, this->indices.size(), tensor.indices.size(), 1, &one,
//                        this_pointer, 1,
//                        that_pointer, 1,
//                        &zero, out_pointer, 1);
//        }

        return raw_output;
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

        const arma::uvec new_table = torque::util::generate_index_table(new_dimension);

        const arma::uword total_elem = this->indices.size();

        thrust::device_vector<T> new_data(total_elem);

        cudaMemcpy(thrust::raw_pointer_cast(new_data.data()),
                   thrust::raw_pointer_cast(this->data.data()),
                   total_elem * sizeof(T),
                   cudaMemcpyDeviceToDevice);

        // It is possible that this tensor has been soft transposed, i.e.
        // the index table may not be in sorted order.
        const arma::uvec sort_index = arma::sort_index(this->index_table);
        arma::uvec new_indices(this->indices.size());

        for(arma::uword i=0; i<total_elem; i++) {

            const arma::uvec old_indices = torque::util::index_to_indices(i, this->index_table, sort_index);
            new_indices(i) = arma::sum(old_indices(permutation) % new_table);

        }

        return SparseTensor<T>(std::move(new_data), gpu::util::arma_to_thrust_device(new_indices), new_table, new_dimension);

    }

protected:
    /// Stores data
    thrust::device_vector<T> data;
};


}
}
#endif //TORQUE_GPU_SPARSE_CUH
