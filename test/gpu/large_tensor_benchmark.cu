#include <catch.hpp>

#define ARMA_ALLOW_FAKE_GCC

#include "tensor/dense.h"
#include "tensor/sparse.h"
#include "tensor/block_sparse.h"

#include "gpu/dense.cuh"
#include "gpu/sparse.cuh"
#include "gpu/block_sparse.cuh"

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


TEST_CASE("Block-sparse n_blocks test") {

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  cublasHandle_t handle;
  cublasCreate(&handle);

  std::cout << "--------- Block-sparse n_blocks test ---------" << std::endl;
  const arma::uvec lengths{4, 8, 16, 32, 64, 128};
  const arma::uvec power{2, 3, 4, 5, 6, 7};
  SECTION("3-rank tensor - matrix contraction") {

    std::cout << "--- 3-rank tensor - matrix contraction ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::BlockSparseTensor<double> tensor(tensor_data.data(),
                                                  arma::uvec{0, 0, 0},
                                                  arma::uvec{0, 0, 0},
                                                  arma::uvec{1, 1, 1});

    torque::gpu::BlockSparseTensor<double> matrix(matrix_data.data(),
                                                  arma::uvec{0, 0},
                                                  arma::uvec{0, 0},
                                                  arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(matrix, {{1, 1}, {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << i << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      const torque::gpu::BlockSparseTensor<double>
          chunk_tensor(tensor_data.data(), arma::uvec{0, 0, 0},
                       arma::uvec{length, length, length} - 1,
                       arma::uvec{length, length, length});

      const torque::gpu::BlockSparseTensor<double>
          chunk_matrix(matrix_data.data(), arma::uvec{0, 0},
                       arma::uvec{length, length} - 1,
                       arma::uvec{length, length});

      for (int j = 0; j <= power(i); j++) {
        auto sliced_tensor = chunk_tensor;
        auto sliced_matrix = chunk_matrix;

        for (int k = 0; k < j; k++) {
          sliced_tensor = sliced_tensor.slice({2, 2, 2});
          sliced_matrix = sliced_matrix.slice({2, 2});
        }

        START_TIMER();
        const auto contraction = sliced_tensor.contract(sliced_matrix,
                                                        {{1, 1},
                                                         {2, 0}});
        STOP_RECORD_TIMER(gpu_time_contraction);

        std::cout << j << "-slice contraction time consumed: " << gpu_time_contraction << std::endl;
      }

    }

  }

  std::cout << std::endl;
  std::cout << "--------- Block-sparse n_blocks test ---------" << std::endl;
}

TEST_CASE("Scaling test") {

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  cublasHandle_t handle;
  cublasCreate(&handle);

  const arma::uvec lengths{1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  SECTION("3-rank tensor - matrix contraction") {

  }


}

