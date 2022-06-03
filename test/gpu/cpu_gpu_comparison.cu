#include <catch.hpp>

#include "tensor/dense.h"
#include "tensor/sparse.h"
#include "tensor/block_sparse.h"
#include "gpu/sparse.cuh"

#define ARMA_ALLOW_FAKE_GCC

TEST_CASE("CPU vs GPU") {

    SECTION("large tensor - matrix contraction") {
        // (From Eric's code)
        cudaEvent_t start;
        cudaEvent_t stop;

        float cpu_time_contraction = -1;
        float gpu_time_contraction = -1;


        cublasHandle_t handle;
        cublasCreate(&handle);

        std::vector<float> tensor_data(24000);
        memset(tensor_data.data(), 0, sizeof(float) * 24000);

        arma::uvec indices(3000);

        for(arma::uword i=0; i<3000; i++) {
            tensor_data[i] = arma::randu();
            indices(i) = i;
        }


        const arma::uvec tensor_dimension{60, 20, 20};
        const arma::uvec tensor_table = torque::util::generate_index_table(tensor_dimension);

        const torque::DenseTensor<float>
                cpu_dense_tensor_format(tensor_data.data(), tensor_dimension);

        const torque::SparseTensor<float> cpu_tensor_format(tensor_data.data(),
                                                            indices,
                                                            tensor_table,
                                                            tensor_dimension);

        const torque::BlockSparseTensor<float> cpu_block_sparse_tensor_format(
                tensor_data.data(),
                arma::uvec{0, 0, 0},
                arma::uvec{29, 9, 9},
                tensor_dimension
                );

        const torque::gpu::SparseTensor<float> tensor_format(tensor_data.data(),
                                                             indices,
                                                             tensor_table,
                                                             tensor_dimension);

        std::vector<float> matrix(1200);
        memset(matrix.data(), 0, sizeof(float) * 1200);
        arma::uvec matrix_indices(300);

        for(arma::uword i=0; i<300; i++) {
            matrix[i] = arma::randu();
            matrix_indices(i) = i;
        }


        const arma::uvec matrix_dimension{60,20};
        const arma::uvec matrix_table = torque::util::generate_index_table(matrix_dimension);

        const torque::DenseTensor<float>
                cpu_dense_matrix_format(matrix.data(), matrix_dimension);
        const torque::SparseTensor<float> cpu_matrix_in_tensor(matrix.data(), matrix_indices, matrix_table, matrix_dimension);
        const torque::BlockSparseTensor<float>
                cpu_block_sparse_matrix_in_tensor(matrix.data(),
                                                  arma::uvec{0, 0},
                                                  arma::uvec{29, 9},
                                                  matrix_dimension);

        const torque::gpu::SparseTensor<float> matrix_in_tensor(matrix.data(), matrix_indices, matrix_table, matrix_dimension);



#define START_TIMER() {               \
      cudaEventCreate(&start);      \
      cudaEventCreate(&stop);       \
      cudaEventRecord(start);       \
    }

#define STOP_RECORD_TIMER(name) {       \
      cudaEventRecord(stop);         \
      cudaEventSynchronize(stop);            \
      cudaEventElapsedTime(&(name), start, stop); \
      cudaEventDestroy(start);                  \
      cudaEventDestroy(stop);                   \
    }

        START_TIMER();

        const auto dense_cpu_contraction =
                cpu_dense_tensor_format.contract(cpu_dense_matrix_format,arma::umat{{0, 0}, {1, 1}});

        STOP_RECORD_TIMER(cpu_time_contraction);

        std::cout <<  "CPU time (dense contraction): " << cpu_time_contraction << " milliseconds" << std::endl;


        START_TIMER();

        const auto cpu_contraction =
                cpu_tensor_format.contract(cpu_matrix_in_tensor,arma::umat{{0, 0}, {1, 1}});

        STOP_RECORD_TIMER(cpu_time_contraction);

        std::cout <<  "CPU time (sparse contraction): " << cpu_time_contraction << " milliseconds" << std::endl;

        START_TIMER();

        const auto cpu_block_sparse_contraction =
                cpu_block_sparse_tensor_format.contract(cpu_block_sparse_matrix_in_tensor,arma::umat{{0, 0}, {1, 1}});

        STOP_RECORD_TIMER(cpu_time_contraction);

        std::cout <<  "CPU time (block sparse contraction): " << cpu_time_contraction << " milliseconds" << std::endl;


        START_TIMER();

        const auto contraction = tensor_format.contract(handle,
                                                        matrix_in_tensor,
                                                        arma::umat{{0, 0}, {1, 1}});

        cublasDestroy(handle);
        STOP_RECORD_TIMER(gpu_time_contraction);

        std::cout <<  "GPU time (sparse contraction): " << gpu_time_contraction << " milliseconds" << std::endl;

    }
}