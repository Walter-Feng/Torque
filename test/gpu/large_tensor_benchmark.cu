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

  cublasHandle_t handle;
  cublasCreate(&handle);

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  std::cout << "--------- Block-sparse n_blocks test ---------" << std::endl;
  const arma::uvec lengths{2, 4, 8, 16, 32, 64};
  const arma::uvec power{0, 2, 3, 4, 4, 4};

  SECTION("Matrix multiplication") {

    std::cout << "--- Matrix multiplication ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::BlockSparseTensor<double> tensor(tensor_data.data(),
                                                  arma::uvec{0, 0},
                                                  arma::uvec{0, 0},
                                                  arma::uvec{1, 1});

    torque::gpu::BlockSparseTensor<double> matrix(matrix_data.data(),
                                                  arma::uvec{0, 0},
                                                  arma::uvec{0, 0},
                                                  arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix, {{1, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      const torque::gpu::BlockSparseTensor<double>
          chunk_tensor(tensor_data.data(), arma::uvec{0, 0},
                       arma::uvec{length, length} - 1,
                       arma::uvec{length, length});

      const torque::gpu::BlockSparseTensor<double>
          chunk_matrix(matrix_data.data(), arma::uvec{0, 0},
                       arma::uvec{length, length} - 1,
                       arma::uvec{length, length});

      for (int j = 0; j <= power(i); j++) {
        auto sliced_tensor = chunk_tensor;
        auto sliced_matrix = chunk_matrix;

        for (int k = 0; k < j; k++) {
          sliced_tensor = sliced_tensor.slice({2, 2});
          sliced_matrix = sliced_matrix.slice({2, 2});
        }

        START_TIMER();
        const auto contraction = sliced_tensor.contract(handle, sliced_matrix,
                                                        {{1, 0}});
        STOP_RECORD_TIMER(gpu_time_contraction);

        std::cout << j << "-slice contraction time consumed: "
                  << gpu_time_contraction << std::endl;
      }

    }

  }

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
    const auto contraction = tensor.contract(handle, matrix, {{1, 1},
                                                              {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

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
        const auto contraction = sliced_tensor.contract(handle, sliced_matrix,
                                                        {{1, 1},
                                                         {2, 0}});
        STOP_RECORD_TIMER(gpu_time_contraction);

        std::cout << j << "-slice contraction time consumed: "
                  << gpu_time_contraction << std::endl;
      }

    }

  }

  SECTION("4-rank tensor - matrix contraction") {

    std::cout << "--- 4-rank tensor - matrix contraction ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::BlockSparseTensor<double> tensor(tensor_data.data(),
                                                  arma::uvec{0, 0, 0, 0},
                                                  arma::uvec{0, 0, 0, 0},
                                                  arma::uvec{1, 1, 1, 1});

    torque::gpu::BlockSparseTensor<double> matrix(matrix_data.data(),
                                                  arma::uvec{0, 0},
                                                  arma::uvec{0, 0},
                                                  arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix, {{1, 1},
                                                              {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length * length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      const torque::gpu::BlockSparseTensor<double>
          chunk_tensor(tensor_data.data(), arma::uvec{0, 0, 0, 0},
                       arma::uvec{length, length, length, length} - 1,
                       arma::uvec{length, length, length, length});

      const torque::gpu::BlockSparseTensor<double>
          chunk_matrix(matrix_data.data(), arma::uvec{0, 0},
                       arma::uvec{length, length} - 1,
                       arma::uvec{length, length});

      for (int j = 0; j <= power(i); j++) {
        auto sliced_tensor = chunk_tensor;
        auto sliced_matrix = chunk_matrix;

        for (int k = 0; k < j; k++) {
          sliced_tensor = sliced_tensor.slice({2, 2, 2, 2});
          sliced_matrix = sliced_matrix.slice({2, 2});
        }

        START_TIMER();
        const auto contraction = sliced_tensor.contract(handle, sliced_matrix,
                                                        {{1, 1},
                                                         {2, 0}});
        STOP_RECORD_TIMER(gpu_time_contraction);

        std::cout << j << "-slice contraction time consumed: "
                  << gpu_time_contraction << std::endl;
      }

    }

  }

  std::cout << std::endl;
  std::cout << "--------- Block-sparse n_blocks test ---------" << std::endl;

  cublasDestroy(handle);
}

TEST_CASE("Dense Tensor Scaling test") {

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  std::cout << "--------- Dense Tensor Scaling test ---------" << std::endl;

  cublasHandle_t handle;
  cublasCreate(&handle);

  const arma::uvec lengths{2, 4, 8, 16, 32, 64};

  SECTION("Matrix multiplication") {

    std::cout << "--- Matrix multiplication ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::DenseTensor<double> tensor(tensor_data.data(),
                                            arma::uvec{1, 1});

    torque::gpu::DenseTensor<double> matrix(matrix_data.data(),
                                            arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix, {{1, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      const torque::gpu::DenseTensor<double>
          chunk_tensor(tensor_data.data(), arma::uvec{length, length});

      const torque::gpu::DenseTensor<double>
          chunk_matrix(matrix_data.data(), arma::uvec{length, length});


      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{1, 0}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      std::cout << "dense contraction time consumed: "
                << gpu_time_contraction << std::endl;


    }

  }

  SECTION("3-rank tensor - matrix contraction") {

    std::cout << "--- 3-rank tensor - matrix contraction ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::DenseTensor<double> tensor(tensor_data.data(),
                                            arma::uvec{1, 1, 1});

    torque::gpu::DenseTensor<double> matrix(matrix_data.data(),
                                            arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix, {{1, 1},
                                                              {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      const torque::gpu::DenseTensor<double>
          chunk_tensor(tensor_data.data(), arma::uvec{length, length, length});

      const torque::gpu::DenseTensor<double>
          chunk_matrix(matrix_data.data(), arma::uvec{length, length});


      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{1, 1},
                                                      {2, 0}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      std::cout << "dense contraction time consumed: "
                << gpu_time_contraction << std::endl;


    }
  }

  SECTION("4-rank tensor - matrix contraction") {

    std::cout << "--- 4-rank tensor - matrix contraction ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::DenseTensor<double> tensor(tensor_data.data(),
                                            arma::uvec{1, 1, 1, 1});

    torque::gpu::DenseTensor<double> matrix(matrix_data.data(),
                                            arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix,
                                             {{1, 1},
                                              {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length * length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      const torque::gpu::DenseTensor<double>
          chunk_tensor(tensor_data.data(),
                       arma::uvec{length, length, length, length});

      const torque::gpu::DenseTensor<double>
          chunk_matrix(matrix_data.data(), arma::uvec{length, length});


      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{1, 1},
                                                      {2, 0}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      std::cout << "dense contraction time consumed: "
                << gpu_time_contraction << std::endl;


    }

  }

  std::cout << std::endl;
  std::cout << "--------- Dense Tensor Scaling test ---------" << std::endl;
  cublasDestroy(handle);

}

TEST_CASE("Sparse Tensor Scaling test") {

  // (From Eric's code)
  cudaEvent_t start;
  cudaEvent_t stop;

  float gpu_time_contraction = -1;

  std::cout << "--------- Sparse Tensor Scaling test ---------" << std::endl;

  cublasHandle_t handle;
  cublasCreate(&handle);

  const arma::uvec lengths{2, 4, 8, 16, 32, 64};
  const arma::uvec power{1, 2, 3, 4, 5, 5};

  SECTION("Matrix multiplication") {

    std::cout << "--- Matrix multiplication ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::SparseTensor<double> tensor(tensor_data.data(),
                                             arma::uvec{0},
                                             arma::uvec{0, 0},
                                             arma::uvec{1, 1});

    torque::gpu::SparseTensor<double> matrix(matrix_data.data(),
                                             arma::uvec{0},
                                             arma::uvec{0, 0},
                                             arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix, {{1, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      arma::uvec tensor_indices(length * length);
      for (arma::uword j = 0; j < tensor_indices.n_elem; j++) {
        tensor_indices(j) = j;
      }
      const arma::uvec tensor_index_table = torque::util::generate_index_table(
          arma::uvec{length, length});

      arma::uvec matrix_indices(length * length);
      for (arma::uword j = 0; j < matrix_indices.n_elem; j++) {
        matrix_indices(j) = j;
      }

      const arma::uvec matrix_index_table = torque::util::generate_index_table(
          arma::uvec{length, length});


      const torque::gpu::SparseTensor<double>
          chunk_tensor(tensor_data.data(), tensor_indices, tensor_index_table,
                       arma::uvec{length, length});

      const torque::gpu::SparseTensor<double>
          chunk_matrix(matrix_data.data(), matrix_indices, matrix_index_table,
                       arma::uvec{length, length});


      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{1, 0}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      std::cout << "sparse contraction time consumed: "
                << gpu_time_contraction << std::endl;


    }

  }

  SECTION("3-rank tensor - matrix contraction") {

    std::cout << "--- 3-rank tensor - matrix contraction ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::SparseTensor<double> tensor(tensor_data.data(),
                                             arma::uvec{0},
                                             arma::uvec{0, 0, 0},
                                             arma::uvec{1, 1, 1});

    torque::gpu::SparseTensor<double> matrix(matrix_data.data(),
                                             arma::uvec{0},
                                             arma::uvec{0, 0},
                                             arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix, {{1, 1},
                                                              {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));


      arma::uvec tensor_indices(length * length * length);
      for (arma::uword j = 0; j < tensor_indices.n_elem; j++) {
        tensor_indices(j) = j;
      }
      const arma::uvec tensor_index_table = torque::util::generate_index_table(
          arma::uvec{length, length, length});

      arma::uvec matrix_indices(length * length);
      for (arma::uword j = 0; j < matrix_indices.n_elem; j++) {
        matrix_indices(j) = j;
      }

      const arma::uvec matrix_index_table = torque::util::generate_index_table(
          arma::uvec{length, length});

      const torque::gpu::SparseTensor<double>
          chunk_tensor(tensor_data.data(), tensor_indices, tensor_index_table,
                       arma::uvec{length, length, length});

      const torque::gpu::SparseTensor<double>
          chunk_matrix(matrix_data.data(), matrix_indices, matrix_index_table,
                       arma::uvec{length, length});


      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{1, 1},
                                                      {2, 0}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      std::cout << "sparse contraction time consumed: "
                << gpu_time_contraction << std::endl;


    }
  }

  SECTION("4-rank tensor - matrix contraction") {

    std::cout << "--- 4-rank tensor - matrix contraction ---" << std::endl;

    std::vector<double> tensor_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    std::vector<double> matrix_data =
        arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(1));

    torque::gpu::SparseTensor<double> tensor(tensor_data.data(),
                                             arma::uvec{0},
                                             arma::uvec{0, 0, 0, 0},
                                             arma::uvec{1, 1, 1, 1});

    torque::gpu::SparseTensor<double> matrix(matrix_data.data(),
                                             arma::uvec{0},
                                             arma::uvec{0, 0},
                                             arma::uvec{1, 1});

    START_TIMER();
    const auto contraction = tensor.contract(handle, matrix,
                                             {{1, 1},
                                              {2, 0}});
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "single element ref: " << gpu_time_contraction << std::endl;

    for (int i = 0; i < lengths.n_elem; i++) {

      std::cout << std::endl;
      std::cout << "contraction with unit length " << lengths(i) << std::endl;

      const arma::uword length = lengths(i);

      std::vector<double> tensor_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length * length * length));

      std::vector<double> matrix_data =
          arma::conv_to<std::vector<double>>::from(
              arma::randu<arma::vec>(length * length));

      arma::uvec tensor_indices(length * length * length * length);
      for (arma::uword j = 0; j < tensor_indices.n_elem; j++) {
        tensor_indices(j) = j;
      }
      const arma::uvec tensor_index_table = torque::util::generate_index_table(
          arma::uvec{length, length, length, length});

      arma::uvec matrix_indices(length * length);
      for (arma::uword j = 0; j < matrix_indices.n_elem; j++) {
        matrix_indices(j) = j;
      }

      const arma::uvec matrix_index_table = torque::util::generate_index_table(
          arma::uvec{length, length});


      const torque::gpu::SparseTensor<double>
          chunk_tensor(tensor_data.data(),
                       tensor_indices,
                       tensor_index_table,
                       arma::uvec{length, length, length, length});

      const torque::gpu::SparseTensor<double>
          chunk_matrix(matrix_data.data(),
                       matrix_indices,
                       matrix_index_table,
                       arma::uvec{length, length});


      START_TIMER();
      const auto contraction = chunk_tensor.contract(handle, chunk_matrix,
                                                     {{1, 1},
                                                      {2, 0}});
      STOP_RECORD_TIMER(gpu_time_contraction);

      std::cout << "sparse contraction time consumed: "
                << gpu_time_contraction << std::endl;


    }

  }

  std::cout << std::endl;
  std::cout << "--------- Sparse Tensor Scaling test ---------" << std::endl;
  cublasDestroy(handle);

}


