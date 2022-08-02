#include <catch.hpp>

#define ARMA_ALLOW_FAKE_GCC

#include "tensor/dense.h"

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


typedef double datatype;

TEST_CASE("Block-sparse n_blocks test") {

#ifdef USE_CUTENSOR
  cutensorHandle_t cutensor_handle;
  cutensorHandle_t * handle = &cutensor_handle;
  cutensorInit(&cutensor_handle);
#else
  cublasHandle_t handle;
  cublasCreate(&handle);
#endif

// (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  std::cout << "--------- Block-sparse n_blocks test ---------" << std::endl;
  const arma::uword length = 32;
  const arma::uword slice = 3;
  const arma::uword n_calls = 1000;

  const double target_sparsity = 0.2;

  double total_time = 0;

  SECTION("4-rank tensor - matrix contraction") {

    std::cout << "--- 4-rank tensor - matrix contraction ---" << std::endl;


    for (int i = 0; i < n_calls; i++) {

      const arma::uword slice_per_dim = std::pow(2, slice);
      const arma::uword block_length_per_dim = length / slice_per_dim;

      const arma::vec rand = arma::randu(std::pow(slice_per_dim, 3));
      const arma::uvec non_trivial_blocks = arma::find(rand < target_sparsity);

      arma::umat begin_points(4, non_trivial_blocks.n_elem);

      const arma::uvec blocks_space{1, slice_per_dim,
                                    slice_per_dim, slice_per_dim};

      const arma::uvec blocks_table = torque::util::generate_index_table(
          blocks_space);

      for (arma::uword j = 0; j < non_trivial_blocks.n_elem; j++) {
        begin_points.col(j) =
            torque::util::index_to_indices(non_trivial_blocks(j),
                                           blocks_table) *
            block_length_per_dim;
      }

      const arma::umat end_points =
          begin_points.each_col() + arma::uvec{31, 3, 3, 3};

      std::vector<datatype> tensor_data =
          arma::conv_to<std::vector<datatype>>::from(
              arma::randu<arma::vec>(
                  std::pow(block_length_per_dim, 3) * 32 *
                  non_trivial_blocks.n_elem));

      const torque::gpu::BlockSparseTensor<datatype>
          chunk_tensor(tensor_data.data(), begin_points, end_points,
                       arma::uvec{length, length, length, length});


      std::vector<datatype> matrix_data =
          arma::conv_to<std::vector<datatype>>::from(
              arma::randu<arma::vec>(32 * 32));

      const arma::vec rand_matrix = arma::randu(8);
      const arma::uvec non_trivial_matrices = arma::find(
          rand_matrix < target_sparsity);
      arma::umat matrix_begin_points(2, non_trivial_matrices.n_elem);
      matrix_begin_points.row(0) = 0;
      matrix_begin_points.row(1) = non_trivial_matrices * 8;

      arma::umat matrix_end_points = matrix_begin_points.each_col() + arma::uvec{31, 3};

      const torque::gpu::BlockSparseTensor<datatype>
          chunk_matrix(matrix_data.data(), matrix_begin_points,
                       matrix_end_points,
                       arma::uvec{length, length});

      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{0, 0},
                                                      {1, 1}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      total_time += gpu_time_contraction;

    }

  }

  std::cout << "contraction time per call: " << total_time << " ms"
            << std::endl;
  std::cout << std::endl;
  std::cout << "--------- Block-sparse n_blocks test ---------" << std::endl;


}