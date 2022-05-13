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
}