#ifndef UTIL_DATA_CONVERSION_H
#define UTIL_DATA_CONVERSION_H

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>

namespace torque {
namespace gpu {
namespace util {

template<typename T, typename U>
void arma_to_cuda(T * dest, const arma::Col<U> & vector) {
  const auto std_vector = arma::conv_to<std::vector<T>>::from(vector);
  cudaMemcpy(dest, std_vector.data(), sizeof(T) * vector.n_elem,
             cudaMemcpyHostToDevice);
}


template<typename T, typename U>
thrust::device_vector <T> arma_to_thrust_device(const arma::Col<U> & vector) {
  const auto std_vector = arma::conv_to<std::vector<T>>::from(vector);

  return thrust::device_vector<T>(std_vector);

}

template<typename T>
arma::uvec
thrust_find(const thrust::device_vector <T> & vector, const arma::uword value) {
  std::vector<arma::uword> result;

  auto begin = vector.begin();

  while (true) {
    auto found = thrust::find(begin, vector.end(), value);

    if (found == vector.end()) {
      break;
    } else {
      result.push_back(found - vector.begin());
      begin = found + 1;
    }

  }

  return result;
}

template<typename T>
void print_cuda(const T * src, const size_t total_length,
                const std::string header = "") {
  std::vector<T> converted(total_length);
  cudaMemcpy(converted.data(), src, sizeof(T) * total_length,
             cudaMemcpyDeviceToHost);
  arma::Col<T>(converted).print(header);

}
}
}
}
#endif //UTIL_DATA_CONVERSION_H
