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


typedef double datatype;

TEST_CASE("Block-sparse n_blocks test") {

#ifdef USE_CUTENSOR
  cutensorHandle_t cutensor_handle;
  cutensorHandle_t * handle = & cutensor_handle;
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
const arma::uvec lengths{32};
const arma::uvec power{4};

SECTION("4-rank tensor - matrix contraction") {

std::cout << "--- 4-rank tensor - matrix contraction ---" << std::endl;

for (int i = 0; i < lengths.n_elem; i++) {

std::cout << std::endl;
std::cout << "contraction with unit length " << lengths(i) << std::endl;

const arma::uword length = lengths(i);

std::vector<datatype> tensor_data =
    arma::conv_to<std::vector<datatype>>::from(
        arma::randu<arma::vec>(length * length * length * length));

std::vector<datatype> matrix_data =
    arma::conv_to<std::vector<datatype>>::from(
        arma::randu<arma::vec>(length * length));

const torque::gpu::BlockSparseTensor<datatype>
    chunk_tensor(tensor_data.data(), arma::uvec{0, 0, 0, 0},
                 arma::uvec{length, length, length, length} - 1,
                 arma::uvec{length, length, length, length});

const torque::gpu::BlockSparseTensor<datatype>
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


}