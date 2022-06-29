#include <catch.hpp>

#define ARMA_ALLOW_FAKE_GCC
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

        std::vector<float> tensor_data(3000);
        std::vector<float> dense_tensor_data(24000);
        memset(tensor_data.data(), 0, sizeof(float) * 3000);
        memset(dense_tensor_data.data(), 0, sizeof(float) * 24000);

        arma::uvec indices(3000);

        const arma::uvec tensor_dimension{60, 20, 20};
        const arma::uvec tensor_table = torque::util::generate_index_table(tensor_dimension);
        const arma::uvec subblock_dimension{30, 10, 10};
        const arma::uvec subblock_table = torque::util::generate_index_table(subblock_dimension);

        for(arma::uword i=0; i<3000; i++) {
            const float rand_number = arma::randu();
            tensor_data[i] = rand_number;

            const arma::uvec original_indices = torque::util::index_to_indices(i, subblock_table);

            const arma::uword original_index = arma::sum(original_indices % tensor_table);

            dense_tensor_data[original_index] = rand_number;
            indices(i) = original_index;
        }


        const torque::DenseTensor<float>
                cpu_dense_tensor_format(dense_tensor_data.data(), tensor_dimension);

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

        std::vector<float> dense_matrix(1200);
        std::vector<float> matrix(300);
        memset(dense_matrix.data(), 0, sizeof(float) * 1200);
        memset(matrix.data(), 0, sizeof(float) * 300);
        arma::uvec matrix_indices(300);

        const arma::uvec matrix_dimension{60,20};
        const arma::uvec matrix_table = torque::util::generate_index_table(matrix_dimension);
        const arma::uvec submatrix_dimension{30, 10};
        const arma::uvec submatrix_table = torque::util::generate_index_table(submatrix_dimension);
        for(arma::uword i=0; i<300; i++) {

            const float rand_number = arma::randu();

            const arma::uvec original_indices = torque::util::index_to_indices(i, submatrix_table);

            const arma::uword original_index = arma::sum(original_indices % matrix_table);

            dense_matrix[original_index] = rand_number;
            matrix_indices(i) = original_index;
            matrix[i] = rand_number;
        }

        const torque::DenseTensor<float>
                cpu_dense_matrix_format(dense_matrix.data(), matrix_dimension);
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

        for(arma::uword i=0; i<10; i++) {
          CHECK(std::abs(dense_cpu_contraction.query({i}) - contraction.query({i})) < 5e-5);
          CHECK(std::abs(cpu_contraction.query({i}) - contraction.query({i})) < 5e-5);
          CHECK(std::abs(cpu_block_sparse_contraction.query({i}) - contraction.query({i})) < 5e-5);
        }
    }
}

