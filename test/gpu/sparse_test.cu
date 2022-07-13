#include <catch.hpp>

#define ARMA_ALLOW_FAKE_GCC

#include "tensor/sparse.h"
#include "gpu/sparse.cuh"


TEST_CASE("thrust - arma utility function") {

  SECTION("vector conversion") {
    arma::uvec test{1, 2, 3, 4, 5};
    const auto converted = torque::gpu::util::arma_to_thrust_device<arma::uword>(
        test);

    CHECK(converted.size() == 5);
    CHECK(converted[0] == 1);
    CHECK(converted[1] == 2);
    CHECK(converted[2] == 3);
  }

  SECTION("thrust find") {
    auto vec = thrust::device_vector<int32_t>(
        std::vector<int32_t>{1, 2, 2, 4, 5});

    CHECK(torque::gpu::util::thrust_find(vec, 2)(0) == 1);
    CHECK(torque::gpu::util::thrust_find(vec, 2)(1) == 2);
    CHECK(torque::gpu::util::thrust_find(vec, 5)(0) == 4);
    CHECK(torque::gpu::util::thrust_find(vec, 6).n_elem == 0);
    vec.push_back(6);
    CHECK(torque::gpu::util::thrust_find(vec, 6)(0) == 5);

  }


}

TEST_CASE("sparse tensor in gpu") {
  SECTION("sparse matrix initialization") {
    const int rank = 2; // Matrix
    const arma::uvec dimension{4, 3}; // 4x3 matrix

    torque::gpu::SparseTensor<float> tensor(dimension);
    const arma::uvec indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    CHECK(tensor.query({1, 1}) == 0);

    //  1  5  9
    //  2  6 10
    //  3  7 11
    //  4  8 12
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    tensor.initialize(a.data(), indices);
    CHECK(tensor.query({1, 2}) == 10);


  }

  SECTION("sparse matrix transposition") {

    const int rank = 2; // Matrix
    const arma::uvec dimension = {4, 3}; // 4x3 matrix

    torque::gpu::SparseTensor<float> tensor(dimension);

    CHECK(tensor.query({1, 1}) == 0);

    //  1  5  9
    //  2  6 10
    //  3  7 11
    //  4  8 12
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const arma::uvec indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    tensor.initialize(a.data(), indices);

    const arma::uvec permutation = {1, 0};

    CHECK(tensor.query({1, 2}) == 10);

    tensor.soft_transpose(permutation); // Now tensor is 3 x 4 matrix

    CHECK(tensor.query({1, 2}) == 7);

    // tensor_transposed should have original matrix, 4 x 3 matrix
    const auto tensor_transposed = tensor.hard_transpose(permutation);

    for (arma::uword i = 0; i < 3; i++) {
      for (arma::uword j = 0; j < 4; j++) {
        CHECK(tensor.query(arma::uvec{i, j}) ==
              tensor_transposed.query(arma::uvec{j, i}));
      }
    }

  }

  SECTION("sparse tensor initialization") {
    const int rank = 3; // Matrix
    const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

    torque::gpu::SparseTensor<float> tensor(dimension);

    CHECK(tensor.query({1, 1, 0}) == 0);

    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const arma::uvec indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    tensor.initialize(a.data(), indices);
    CHECK(tensor.query({0, 1, 2}) == 11);
  }

  SECTION("sparse tensor transposition") {
    const int rank = 3; // Matrix
    const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const arma::uvec indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    torque::gpu::SparseTensor<float> tensor(a.data(), indices,
                                            torque::util::generate_index_table(
                                                dimension),
                                            dimension);

    CHECK(tensor.query({0, 0, 2}) == 9);
    CHECK(tensor.query({0, 1, 2}) == 11);
    tensor.soft_transpose({0, 2, 1});
    CHECK(tensor.query({0, 2, 0}) == 9);
    CHECK(tensor.query({0, 2, 1}) == 11);

    const auto tensor_another_transpose = tensor.hard_transpose({1, 0, 2});
    CHECK(tensor_another_transpose.query({2, 0, 0}) == 9);
    CHECK(tensor_another_transpose.query({2, 0, 1}) == 11);
  }

  SECTION("scalar") {

    torque::gpu::SparseTensor<float> tensor(arma::uvec{});

    CHECK(tensor.rank == 0);
    CHECK(tensor.query({}) == 0);
    tensor.modify({}, 1);
    CHECK(tensor.query({}) == 1);

  }

  SECTION("vector contraction") {

    cublasHandle_t handle;
    cublasCreate(&handle);

    const std::vector<float> vec{0, 1, 2, 3, 4};
    const arma::uvec indices{0, 1, 2, 3, 4};
    const arma::uvec dimension{5};

    const torque::gpu::SparseTensor<float> tensor_format(vec.data(), indices,
                                                         torque::util::generate_index_table(
                                                             dimension),
                                                         dimension);
    const auto result =
        tensor_format.contract(handle, tensor_format, arma::umat({0, 0}));

    assert(result.to_number() == 30);
    cublasDestroy(handle);

  }

  SECTION("matrix multiplication") {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const std::vector<float> vec{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const arma::uvec indices{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const arma::uvec dimension{3, 3};
    const arma::uvec index_table = torque::util::generate_index_table(
        dimension);

    const torque::gpu::SparseTensor<float> tensor_format(vec.data(), indices,
                                                         index_table, {3, 3});

    const auto A_squared = tensor_format.contract(handle, tensor_format,
                                                  arma::umat({1, 0}));

    const arma::mat ref_A_squared{{15, 42, 69},
                                  {18, 54, 90},
                                  {21, 66, 111}};

    for (arma::uword i = 0; i < 3; i++) {
      for (arma::uword j = 0; j < 3; j++) {
        CHECK(A_squared.query(arma::uvec{i, j}) == ref_A_squared(i, j));
      }
    }

    cublasDestroy(handle);
  }

  SECTION("matrix inner product") {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const std::vector<float> vec{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const arma::uvec indices{0, 1, 2, 3, 4, 5, 6, 7, 8};
    const arma::uvec dimension{3, 3};
    const arma::uvec index_table = torque::util::generate_index_table(
        dimension);

    const torque::gpu::SparseTensor<float> tensor_format(vec.data(), indices,
                                                         index_table,
                                                         dimension);

    const auto A_squared =
        tensor_format.contract(handle, tensor_format,
                               arma::umat({{1, 0},
                                           {0, 1}}));

    const arma::mat ref_A = arma::mat{{0, 1, 2},
                                      {3, 4, 5},
                                      {6, 7, 8}}.t();

    const double result = arma::accu(ref_A % ref_A.t());

    CHECK(A_squared.to_number() == result);

    cublasDestroy(handle);
  }

  SECTION("tensor-vector contraction") {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const std::vector<float> tensor_data{0, 1, 2, 3, 4, 5, 6, 7};
    const arma::uvec indices{0, 1, 2, 3, 4, 5, 6, 7};
    const arma::uvec tensor_dimension{2, 2, 2};
    const arma::uvec tensor_table = torque::util::generate_index_table(
        tensor_dimension);

    const torque::gpu::SparseTensor<float> tensor_format(tensor_data.data(),
                                                         indices,
                                                         tensor_table,
                                                         tensor_dimension);


    const std::vector<float> vector{1, 2};
    const arma::uvec vector_indices{0, 1};
    const arma::uvec vector_dimension{2};
    const arma::uvec vector_table = torque::util::generate_index_table(
        vector_dimension);

    const torque::gpu::SparseTensor<float> vector_in_tensor(vector.data(),
                                                            vector_indices,
                                                            vector_table,
                                                            {2});

    const auto contraction = tensor_format.contract(handle, vector_in_tensor,
                                                    arma::umat{{0, 0}});

    CHECK(contraction.query({0, 0}) == 2);
    CHECK(contraction.query({1, 0}) == 8);
    CHECK(contraction.query({0, 1}) == 14);
    CHECK(contraction.query({1, 1}) == 20);

    const auto contraction_2 = tensor_format.contract(handle, vector_in_tensor,
                                                      arma::umat{{1, 0}});

    CHECK(contraction_2.query({0, 0}) == 4);
    CHECK(contraction_2.query({1, 0}) == 7);
    CHECK(contraction_2.query({0, 1}) == 16);
    CHECK(contraction_2.query({1, 1}) == 19);

    const auto contraction_3 = tensor_format.contract(handle, vector_in_tensor,
                                                      arma::umat{{2, 0}});

    CHECK(contraction_3.query({1, 0}) == 11);

    cublasDestroy(handle);

  }

  SECTION("tensor-matrix contraction") {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const std::vector<float> tensor_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const arma::uvec indices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};


    const arma::uvec tensor_dimension{2, 2, 3};
    const arma::uvec tensor_table = torque::util::generate_index_table(
        tensor_dimension);

    const torque::gpu::SparseTensor<float> tensor_format(tensor_data.data(),
                                                         indices,
                                                         tensor_table,
                                                         tensor_dimension);


    const std::vector<float> matrix{1, 2, 3, 4};
    const arma::uvec matrix_indices{0, 1, 2, 3};

    const arma::uvec matrix_dimension{2, 2};
    const arma::uvec matrix_table = torque::util::generate_index_table(
        matrix_dimension);

    const torque::gpu::SparseTensor<float> matrix_in_tensor(matrix.data(),
                                                            matrix_indices,
                                                            matrix_table,
                                                            matrix_dimension); // row vector

    const auto contraction = tensor_format.contract(handle, matrix_in_tensor,
                                                    arma::umat{{0, 1},
                                                               {1, 0}});

    CHECK(contraction.query({0}) == 19);
    CHECK(contraction.query({1}) == 59);
    CHECK(contraction.query({2}) == 99);

    cublasDestroy(handle);

  }

  SECTION("large tensor contraction") {
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::vector<float> tensor_data(3000);
    arma::uvec indices(3000);

    for (arma::uword i = 0; i < 3000; i++) {
      tensor_data[i] = arma::randu();
      indices(i) = i;
    }


    const arma::uvec tensor_dimension{30, 10, 10};
    const arma::uvec tensor_table = torque::util::generate_index_table(
        tensor_dimension);

    const torque::gpu::SparseTensor<float> tensor_format(tensor_data.data(),
                                                         indices,
                                                         tensor_table,
                                                         tensor_dimension);

    std::vector<float> matrix(300);
    arma::uvec matrix_indices(300);

    for (arma::uword i = 0; i < 300; i++) {
      matrix[i] = arma::randu();
      matrix_indices(i) = i;
    }


    const arma::uvec matrix_dimension{30, 10};
    const arma::uvec matrix_table = torque::util::generate_index_table(
        matrix_dimension);

    const torque::gpu::SparseTensor<float> matrix_in_tensor(matrix.data(),
                                                            matrix_indices,
                                                            matrix_table,
                                                            matrix_dimension); // row vector

// (From Eric's code)
    cudaEvent_t start;
    cudaEvent_t stop;

    float gpu_time_contraction = -1;

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

    const auto contraction = tensor_format.contract(handle,
                                                    matrix_in_tensor,
                                                    arma::umat{{0, 0},
                                                               {1, 1}});

    cublasDestroy(handle);
    STOP_RECORD_TIMER(gpu_time_contraction);

    std::cout << "GPU time (contraction): " << gpu_time_contraction
              << " milliseconds" << std::endl;
  }

}