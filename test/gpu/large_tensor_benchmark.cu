#include <catch.hpp>

#include "tensor/dense.h"
#include "tensor/sparse.h"
#include "tensor/block_sparse.h"

#include "gpu/dense.cuh"
#include "gpu/sparse.cuh"
#include "gpu/block_sparse.cuh"

template<typename T>
torque::gpu::BlockSparseTensor<T>
slice(const torque::gpu::BlockSparseTensor<T> & tensor,
      const arma::uvec & divisor) {


}

TEST_CASE("Scaling test") {

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  cublasHandle_t handle;
  cublasCreate(&handle);

  const arma::uvec lengths{1, 16, 32, 48, 64, 80, 96, 128};
  SECTION("3-rank tensor - matrix contraction") {

    for (int i = 0; i < lengths.n_elem; i++) {

    }

  }


}

TEST_CASE("Block-sparse n_blocks test") {

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

