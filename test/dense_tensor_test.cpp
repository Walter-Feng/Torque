#include <catch.hpp>

#include "tensor/dense.h"

TEST_CASE("dense tensor operation") {
    SECTION("dense matrix initialization") {
        const int rank = 2; // Matrix
        const arma::uvec dimension = {4, 3}; // 4x3 matrix

        torque::DenseTensor<float> tensor(dimension);

        CHECK(tensor.query({1,1}) == 0);

        //  1  5  9
        //  2  6 10
        //  3  7 11
        //  4  8 12
        std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        tensor.initialize(a.data());
        CHECK(tensor.query({1,2}) == 10);
    }

    SECTION("dense matrix transposition") {

        const int rank = 2; // Matrix
        const arma::uvec dimension = {4, 3}; // 4x3 matrix

        torque::DenseTensor<float> tensor(dimension);

        CHECK(tensor.query({1,1}) == 0);

        //  1  5  9
        //  2  6 10
        //  3  7 11
        //  4  8 12
        std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        tensor.initialize(a.data());

        const arma::uvec permutation = {1,0};

        CHECK(tensor.query({1,2}) == 10);

        tensor.soft_transpose(permutation); // Now tensor is 3 x 4 matrix

        CHECK(tensor.query({1, 2}) == 7);

        // tensor_transposed should have original matrix, 4 x 3 matrix
        const auto tensor_transposed = tensor.hard_transpose(permutation);

        for(arma::uword i=0; i<3; i++) {
            for(arma::uword j=0; j<4; j++) {
                CHECK(tensor.query(arma::uvec{i, j}) == tensor_transposed.query(arma::uvec{j, i}));
            }
        }

    }

    SECTION("dense tensor initialization") {
        const int rank = 3; // Matrix
        const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

        torque::DenseTensor<float> tensor(dimension);

        CHECK(tensor.query({1,1,0}) == 0);

        std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        tensor.initialize(a.data());
        CHECK(tensor.query({0,1,2}) == 11);
    }

    SECTION("dense tensor transposition") {
        const int rank = 3; // Matrix
        const arma::uvec dimension = {2, 2, 3}; // 4x3 matrix

        std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        torque::DenseTensor<float> tensor(a.data(), dimension);
        CHECK(tensor.query({0,0,2}) == 9);
        CHECK(tensor.query({0,1,2}) == 11);
        tensor.soft_transpose({0, 2, 1});
        CHECK(tensor.query({0,2,0}) == 9);
        CHECK(tensor.query({0,2,1}) == 11);

        const auto tensor_another_transpose = tensor.hard_transpose({1,0,2});
        CHECK(tensor_another_transpose.query({2, 0, 0}) == 9);
        CHECK(tensor_another_transpose.query({2, 0, 1}) == 11);
    }
}