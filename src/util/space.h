#ifndef TORQUE_UTIL_SPACE_H
#define TORQUE_UTIL_SPACE_H

#include <armadillo>
#include "error.h"

namespace torque {
namespace util {

arma::uvec generate_index_table(const arma::uvec & dimension);
arma::uvec index_to_indices(arma::uword i, const arma::uvec & table);
arma::uvec index_to_indices(arma::uword i, const arma::uvec & table, const arma::uvec & sort_index);

template<typename T>
arma::Col<T> nest_sum(const arma::Col<T> & summed_list, const T initial = 0) {
    arma::uvec result = arma::uvec(arma::size(summed_list), arma::fill::zeros);

    T temp = initial;

    for(arma::uword i=0; i<result.n_elem; i++) {
        result(i) = temp;
        temp += summed_list(i);
    }

    return result;
}

arma::uvec in_range(const arma::uvec & indices, const arma::umat & begin_points, const arma::umat & end_points);
}
}
#endif //TORQUE_UTIL_SPACE_H
