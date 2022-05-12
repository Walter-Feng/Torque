#ifndef TORQUE_DENSE_H
#define TORQUE_DENSE_H

#include "tensor.h"
#include "error.h"

namespace torque {
    template<typename T>
    class DenseTensor : Tensor<T>
    {
    public:
        void initialize(const arma::uvec & dimension) {
            if(this->data) {
                free(this->data);
            }

            this->data = (T *) malloc(sizeof(T) * arma::prod(dimension));
            memset(this->data, 0, sizeof(T) * arma::prod(dimension));

        }

        void initialize(const T * source_data, const arma::uvec & dimension) {
            if(source_data) {
                memcpy(this->data, source_data, sizeof(T) * arma::prod(dimension));
            } else {
                throw Error("Source data not allocated!");
            }
        }

        ~DenseTensor() {
            if(this->data) {
                free(this->data);
            }
        }

        T query(const arma::uvec & indices) const {
            if(this->data) {

                return this->data[arma::sum(indices * this->index_table)];

            } else {
                throw Error("Tensor not initialized");
            }
        }
    };

}

#endif //TORQUE_DENSE_H
