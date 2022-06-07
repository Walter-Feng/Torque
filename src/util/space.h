#ifndef TORQUE_UTIL_SPACE_H
#define TORQUE_UTIL_SPACE_H

#include <armadillo>
#include "error.h"

namespace torque {
namespace util {

inline
arma::uvec generate_index_table(const arma::uvec & dimension) {

    arma::uvec table(dimension.n_elem);

    arma::uword table_index = 1;

    for (arma::uword i = 0; i < dimension.n_elem; i++) {
        table(i) = table_index;
        table_index *= dimension(i);
    }

    return table;

}

inline
arma::uvec index_to_indices(const arma::uword i, const arma::uvec & table) {

    arma::uword temp_i = i;
    arma::uvec indices(table.n_elem);

    assert(table.is_sorted());

    for(arma::uword j = table.n_elem - 1; j > 0; j--) {
        indices(j) = temp_i / table(j);
        temp_i = temp_i % table(j);
    }

    indices(0) = temp_i;

    return indices;
}

inline
arma::uvec index_to_indices(const arma::uword i, const arma::uvec & table, const arma::uvec & sort_index) {

    arma::uword temp_i = i;
    arma::uvec indices(table.n_elem);

    for(arma::uword j = table.n_elem - 1; j > 0; j--) {
        indices(sort_index(j)) = temp_i / table(sort_index(j));
        temp_i %= table(sort_index(j));
    }

    indices(sort_index(0)) = temp_i;

    return indices;
}

inline
arma::uvec in_range(const arma::uvec & indices, const arma::umat & begin_points, const arma::umat & end_points) {

    assert(begin_points.n_cols == end_points.n_cols);

    arma::Col<int> true_false_list(begin_points.n_cols);

    for(arma::uword i=0; i<begin_points.n_cols; i++) {
        true_false_list(i) = arma::all(indices >= begin_points.col(i))
                             && arma::all(indices <= end_points.col(i));
    }
    return arma::find(true_false_list != 0);
}

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


}
}
#endif //TORQUE_UTIL_SPACE_H
