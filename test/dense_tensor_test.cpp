#include <catch.hpp>

#include "tensor/dense.h"

TEST_CASE("dense tensor operation") {
  SECTION("dense matrix initialization") {
    const int rank = 2; // Matrix
    const arma::uvec dimension = {4, 3}; // 4x3 matrix

    torque::DenseTensor<float> tensor(dimension);

    CHECK(tensor.query({1, 1}) == 0);

    //  1  5  9
    //  2  6 10
    //  3  7 11
    //  4  8 12
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    tensor.initialize(a.data());
    CHECK(tensor.query({1, 2}) == 10);
  }

  SECTION("dense matrix transposition") {

    const int rank = 2; // Matrix
    const arma::uvec dimension = {4, 3}; // 4x3 matrix

    torque::DenseTensor<float> tensor(dimension);

    CHECK(tensor.query({1, 1}) == 0);

    //  1  5  9
    //  2  6 10
    //  3  7 11
    //  4  8 12
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    tensor.initialize(a.data());

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

  SECTION("dense tensor initialization") {
    const int rank = 3; // Matrix
    const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

    torque::DenseTensor<float> tensor(dimension);

    CHECK(tensor.query({1, 1, 0}) == 0);

    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    tensor.initialize(a.data());
    CHECK(tensor.query({0, 1, 2}) == 11);
  }

  SECTION("dense tensor transposition") {
    const int rank = 3; // Matrix
    const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    torque::DenseTensor<float> tensor(a.data(), dimension);
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

    torque::DenseTensor<float> tensor(arma::uvec{});

    CHECK(tensor.rank == 0);
    CHECK(tensor.query({}) == 0);
    tensor.modify({}, 1);
    CHECK(tensor.query({}) == 1);

  }

  SECTION("vector contraction") {
    const std::vector<float> vec{0, 1, 2, 3, 4};

    const torque::DenseTensor<float> tensor_format(vec.data(), {5});
    const auto result =
        tensor_format.contract(tensor_format, arma::umat({0, 0}));

    assert(result.to_number() == 30);
  }

  SECTION("matrix multiplication") {
    const std::vector<float> vec{0, 1, 2, 3, 4, 5, 6, 7, 8};

    const torque::DenseTensor<float> tensor_format(vec.data(), {3, 3});

    const auto A_squared = tensor_format.contract(tensor_format,
                                                  arma::umat({1, 0}));

    const arma::mat ref_A = arma::mat{{0, 1, 2},
                                      {3, 4, 5},
                                      {6, 7, 8}}.t();

    const arma::mat ref_A_squared = ref_A * ref_A;

    for (arma::uword i = 0; i < 3; i++) {
      for (arma::uword j = 0; j < 3; j++) {
        CHECK(A_squared.query(arma::uvec{i, j}) == ref_A_squared(i, j));
      }
    }
  }

  SECTION("matrix inner product") {
    const std::vector<float> vec{0, 1, 2, 3, 4, 5, 6, 7, 8};

    const torque::DenseTensor<float> tensor_format(vec.data(), {3, 3});

    const auto A_squared =
        tensor_format.contract(tensor_format,
                               arma::umat({{1, 0},
                                           {0, 1}}));

    const arma::mat ref_A = arma::mat{{0, 1, 2},
                                      {3, 4, 5},
                                      {6, 7, 8}}.t();

    const double result = arma::accu(ref_A % ref_A.t());

    CHECK(A_squared.to_number() == result);
  }

  SECTION("tensor-vector contraction") {
    const std::vector<float> tensor_data{0, 1, 2, 3, 4, 5, 6, 7};

    const torque::DenseTensor<float> tensor_format(tensor_data.data(),
                                                   {2, 2, 2});

    const std::vector<float> vector{1, 2};

    const torque::DenseTensor<float> vector_in_tensor(vector.data(), {2});

    const auto contraction = tensor_format.contract(vector_in_tensor,
                                                    arma::umat{{0, 0}});

    CHECK(contraction.query({0, 0}) == 2);
    CHECK(contraction.query({1, 0}) == 8);
    CHECK(contraction.query({0, 1}) == 14);
    CHECK(contraction.query({1, 1}) == 20);

    const auto contraction_2 = tensor_format.contract(vector_in_tensor,
                                                      arma::umat{{1, 0}});

    CHECK(contraction_2.query({0, 0}) == 4);
    CHECK(contraction_2.query({1, 0}) == 7);
    CHECK(contraction_2.query({0, 1}) == 16);
    CHECK(contraction_2.query({1, 1}) == 19);

    const auto contraction_3 = tensor_format.contract(vector_in_tensor,
                                                      arma::umat{{2, 0}});

    CHECK(contraction_3.query({1, 0}) == 11);

  }

  SECTION("tensor-matrix contraction") {
    const std::vector<float> tensor_data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    const torque::DenseTensor<float> tensor_format(tensor_data.data(),
                                                   {2, 2, 3});

    const std::vector<float> matrix{1, 2, 3, 4};

    const torque::DenseTensor<float> matrix_in_tensor(matrix.data(),
                                                      {2, 2}); // row vector

    const auto contraction = tensor_format.contract(matrix_in_tensor,
                                                    arma::umat{{0, 1},
                                                               {1, 0}});

    CHECK(contraction.query({0}) == 19);
    CHECK(contraction.query({1}) == 59);
    CHECK(contraction.query({2}) == 99);
  }

}