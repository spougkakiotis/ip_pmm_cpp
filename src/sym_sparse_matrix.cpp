#include "sym_sparse_matrix.hpp"

#include <cassert>
#include <stdexcept>
#include <utility>

namespace ippmm {

    SymSparseMatrix::SymSparseMatrix(Int n,
                                    std::vector<Int> col_ptr,
                                    std::vector<Int> row_idx,
                                    std::vector<Scalar> values)
        : upper_(n, n, std::move(col_ptr), std::move(row_idx), std::move(values)) {
        // Enforce the upper-triangular contract (external input -> throw).
        for (Int j = 0; j < upper_.cols(); ++j) {
            for (Int k = upper_.col_ptr()[j]; k < upper_.col_ptr()[j + 1]; ++k) {
                if (upper_.row_idx()[k] > j) {
                    throw std::invalid_argument(
                        "SymSparseMatrix: storage must be upper-triangular (row > col found)");
                }
            }
        }
    }

    std::vector<Scalar> SymSparseMatrix::multiply(const std::vector<Scalar>& x) const {
        assert(static_cast<Int>(x.size()) == size());

        std::vector<Scalar> y(size(), 0.0);
        const auto& Ap = upper_.col_ptr();
        const auto& Ai = upper_.row_idx();
        const auto& Ax = upper_.values();

        for (Int j = 0; j < size(); ++j) {
            for (Int k = Ap[j]; k < Ap[j + 1]; ++k) {
                const Int    i = Ai[k];
                const Scalar v = Ax[k];
                y[i] += v * x[j];          // upper entry
                if (i != j) y[j] += v * x[i];  // mirrored lower entry
            }
        }
        return y;
    }

}  // namespace ippmm