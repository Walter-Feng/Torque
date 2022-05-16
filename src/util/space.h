#ifndef TORQUE_UTIL_SPACE_H
#define TORQUE_UTIL_SPACE_H

#include <armadillo>
#include "error.h"

namespace torque {
namespace util {

arma::uvec generate_index_table(const arma::uvec & dimension);
arma::uvec index_to_indices(const arma::uword i, const arma::uvec & table);

}
}
#endif //TORQUE_UTIL_SPACE_H
