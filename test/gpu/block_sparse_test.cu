#include <catch.hpp>

#include "gpu/block_sparse.cuh"

TEST_CASE("block sparse tensor operation") {

#ifdef USE_CUTENSOR
  cutensorHandle_t cutensor_handle;
  cutensorHandle_t * handle = &cutensor_handle;
  cutensorInit(handle);
#else
  cublasHandle_t handle;
  cublasCreate(&handle);
#endif

  SECTION("block sparse matrix initialization") {
    const int rank = 2; // Matrix
    std::vector<float> a = {1, 2, 3, 4}; // 2x2 sub block

    const arma::uvec dimension = {4, 3}; // 4x3 matrix

    torque::gpu::BlockSparseTensor<float> tensor(a.data(), arma::uvec{1, 1},
                                                 arma::uvec{2, 2}, dimension);

    CHECK(tensor.query({1, 1}) == 1);
    CHECK(tensor.query({2, 2}) == 4);
    CHECK(tensor.query({3, 2}) == 0);

    tensor.modify({2, 2}, 0);
    CHECK(tensor.query({2, 2}) == 0);


    tensor.modify({3, 2}, 5);
    CHECK(tensor.query({3, 2}) == 5);

    tensor.append_block(a.data(), {0, 0}, {1, 1}, {1, 2});
    CHECK(tensor.query({1, 1}) == 5);

  }

  SECTION("block sparse matrix transposition") {
    const int rank = 2; // Matrix
    std::vector<float> a = {1, 2, 3, 4}; // 2x2 sub block

    const arma::uvec dimension = {4, 3}; // 4x3 matrix

    // 0 0 0
    // 0 0 0
    // 1 3 0
    // 2 4 0
    torque::gpu::BlockSparseTensor<float> tensor(a.data(), arma::uvec{2, 0},
                                                 arma::uvec{3, 1}, dimension);

    // 0 0 1 2
    // 0 0 3 4
    // 0 0 0 0
    tensor = tensor.hard_transpose({1, 0});

    CHECK(tensor.query({0, 3}) == 2);
    CHECK(tensor.query({1, 2}) == 3);

    const auto original_tensor = tensor.hard_transpose({1, 0});

    CHECK(original_tensor.query({3, 0}) == 2);
    CHECK(original_tensor.query({2, 1}) == 3);
  }

  SECTION("block sparse tensor initialization") {
    const int rank = 3;
    const arma::uvec dimension = {2, 2, 3};

    torque::gpu::BlockSparseTensor<float> tensor(dimension);

    CHECK(tensor.query({1, 1, 0}) == 0);

    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    tensor.append_block(a.data(), {0, 0, 0}, {1, 1, 2}, {1, 2, 4});
    CHECK(tensor.query({0, 1, 2}) == 11);
  }

  SECTION("block sparse tensor transposition") {
    const int rank = 3; // Matrix
    const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

    std::vector<float> a = {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0};
    std::vector<float> b = {7, 8, 9, 10, 11, 12};

    torque::gpu::BlockSparseTensor<float> tensor(dimension);
    tensor.append_block(a.data(), {0, 0, 0}, {1, 1, 2}, {1, 2, 4});
    tensor.append_block(b.data(), {1, 0, 0}, {1, 1, 2}, {1, 1, 2});

    const auto tensor_another_transpose = tensor.hard_transpose({1, 0, 2});
    CHECK(tensor_another_transpose.query({1, 0, 0}) == 3);
    CHECK(tensor_another_transpose.query({1, 0, 1}) == 0);
    CHECK(tensor_another_transpose.query({1, 1, 2}) == 12);
  }

  SECTION("scalar") {

    torque::gpu::BlockSparseTensor<float> tensor(arma::uvec{});

    CHECK(tensor.rank == 0);
    CHECK(tensor.query({}) == 0);
    tensor.modify({}, 1);
    CHECK(tensor.query({}) == 1);

  }

  SECTION("vector contraction") {
    const std::vector<float> vec1{0, 1};
    const std::vector<float> vec2{2, 3, 4};

    torque::gpu::BlockSparseTensor<float> tensor_format({5});
    tensor_format.append_block(vec1.data(), {0}, {1}, {1});
    tensor_format.append_block(vec2.data(), {2}, {4}, {1});

    const auto result =
        tensor_format.contract(handle, tensor_format,
                               arma::umat({{0, 0}}));

    CHECK(result.to_number() == 30);
  }

  SECTION("matrix multiplication") {
    const std::vector<float> vec{0, 1, 0, 0, 2, 3, 0, 0, 0, 0, 4, 5, 0, 0, 6,
                                 7};

    const std::vector<float> block_1{0, 1, 2, 3};
    const std::vector<float> block_2{4, 5, 6, 7};

    const torque::DenseTensor<float> tensor_format(vec.data(), {4, 4});
    torque::gpu::BlockSparseTensor<float> sparse_tensor_format({4, 4});
    sparse_tensor_format.append_block(block_1.data(), {0, 0}, {1, 1}, {1, 2});
    sparse_tensor_format.append_block(block_2.data(), {2, 2}, {3, 3}, {1, 2});

    const auto A_squared = tensor_format.contract(tensor_format,
                                                  arma::umat({1, 0}));
    const auto A_squared_sparse = sparse_tensor_format.contract(
        handle, sparse_tensor_format, arma::umat({{1, 0}}));

    for (arma::uword i = 0; i < 4; i++) {
      for (arma::uword j = 0; j < 4; j++) {
        CHECK(A_squared.query(arma::uvec{i, j}) ==
              A_squared_sparse.query({i, j}));
      }
    }
  }

  SECTION("matrix inner product") {
    const std::vector<float> block1{0, 1, 2, 3};

    const std::vector<float> block2{0, 7, 8, 9};

    torque::gpu::BlockSparseTensor<float> tensor_format({3, 3});

    tensor_format.append_block(block1.data(), {0, 0}, {1, 1}, {1, 2});
    tensor_format.append_block(block2.data(), {1, 1}, {2, 2}, {1, 2});

    const auto A_squared =
        tensor_format.contract(handle, tensor_format,
                               arma::umat({{1, 0},
                                           {0, 1}}));

    const arma::mat ref_A = arma::mat{{0, 1, 0},
                                      {2, 3, 7},
                                      {0, 8, 9}}.t();

    const double result = arma::accu(ref_A % ref_A.t());

    CHECK(A_squared.to_number() == result);
  }

  SECTION("tensor-vector contraction") {
    const std::vector<float> tensor_data_1{0, 1, 2, 3};
    const std::vector<float> tensor_data_2{4, 5, 6, 7};

    torque::gpu::BlockSparseTensor<float> tensor_format({2, 2, 2});

    tensor_format.append_block(tensor_data_1.data(), {0, 0, 0}, {1, 1, 0},
                               {1, 2, 4});
    tensor_format.append_block(tensor_data_2.data(), {0, 0, 1}, {1, 1, 1},
                               {1, 2, 4});

    const std::vector<float> vector{1, 2};

    torque::gpu::BlockSparseTensor<float> vector_in_tensor({2});
    vector_in_tensor.append_block(vector.data(), {0}, {1}, {1});

    const auto contraction = tensor_format.contract(handle,
                                                    vector_in_tensor,
                                                    arma::umat{{0, 0}});

    CHECK(contraction.query({0, 0}) == 2);
    CHECK(contraction.query({1, 0}) == 8);
    CHECK(contraction.query({0, 1}) == 14);
    CHECK(contraction.query({1, 1}) == 20);

    const auto contraction_2 = tensor_format.contract(handle,
                                                      vector_in_tensor,
                                                      arma::umat{{1, 0}});

    CHECK(contraction_2.query({0, 0}) == 4);
    CHECK(contraction_2.query({1, 0}) == 7);
    CHECK(contraction_2.query({0, 1}) == 16);
    CHECK(contraction_2.query({1, 1}) == 19);

    const auto contraction_3 = tensor_format.contract(handle,
                                                      vector_in_tensor,
                                                      arma::umat{{2, 0}});

    const arma::uword n_elem = arma::sum(contraction_3.block_n_elem);
    std::vector<float> check(n_elem);
    cudaMemcpy(check.data(), contraction_3.data, n_elem * sizeof(float), cudaMemcpyDeviceToHost);
    arma::Col<float>(check).print("data");

    CHECK(contraction_3.query({1, 0}) == 11);

  }

  SECTION("tensor-matrix contraction") {
    const std::vector<float> tensor_data1{0, 1, 2, 3};
    const std::vector<float> tensor_data2{4, 5, 6, 7};
    const std::vector<float> tensor_data3{8, 9, 10, 11};

    torque::gpu::BlockSparseTensor<float> tensor_format({2, 2, 3});
    tensor_format.append_block(tensor_data3.data(), {0, 0, 2}, {1, 1, 2},
                               {1, 2, 4});
    tensor_format.append_block(tensor_data2.data(), {0, 0, 1}, {1, 1, 1},
                               {1, 2, 4});
    tensor_format.append_block(tensor_data1.data(), {0, 0, 0}, {1, 1, 0},
                               {1, 2, 4});

    const std::vector<float> matrix{1, 2, 3, 4};

    torque::gpu::BlockSparseTensor<float> matrix_in_tensor(
        {2, 2}); // row vector

    matrix_in_tensor.append_block(matrix.data(), {0, 0}, {1, 1}, {1, 2});
    const auto contraction = tensor_format.contract(handle,
                                                    matrix_in_tensor,
                                                    arma::umat{{0, 1},
                                                               {1, 0}});

    CHECK(contraction.query({0}) == 19);
    CHECK(contraction.query({1}) == 59);
    CHECK(contraction.query({2}) == 99);
  }

#ifndef USE_CUTENSOR
  cublasDestroy(handle);
#endif

  cudaDeviceSynchronize();
  cudaDeviceReset();
  cudaDeviceSynchronize();
}