#include <catch.hpp>

#define ARMA_ALLOW_FAKE_GCC

#include "tensor/dense.h"
#include "tensor/sparse.h"
#include "tensor/block_sparse.h"

#include "gpu/dense.cuh"
#include "gpu/sparse.cuh"
#include "gpu/block_sparse.cuh"


TEST_CASE("Block-sparse n_blocks test") {

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  cublasHandle_t handle;
  cublasCreate(&handle);

  const arma::uvec lengths{1, 16, 32, 48, 64, 80, 96, 128};
  SECTION("3-rank tensor - matrix contraction") {

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));


    const arma::umat begin_point = arma::uvec{0, 0, 0};
    const arma::umat end_point = arma::uvec{0, 0, 0};

    torque::gpu::BlockSparseTensor<double> chunk_tensor(tensor_data.data(),
                                                        arma::uvec{0, 0, 0},
                                                        arma::uvec{0, 0, 0},
                                                        arma::uvec{1, 1, 1});

    torque::gpu::BlockSparseTensor<double> chunk_matrix(matrix_data.data(),
                                                        arma::uvec{0, 0},
                                                        arma::uvec{0, 0},
                                                        arma::uvec{1, 1});

    const auto contraction = chunk_tensor.contract(chunk_matrix, {{1, 1}, {2, 0}});

    CHECK(contraction.query({0}) == tensor_data.at(0) * matrix_data.at(0));

    for (int i = 0; i < lengths.n_elem; i++) {
    }

  }


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

