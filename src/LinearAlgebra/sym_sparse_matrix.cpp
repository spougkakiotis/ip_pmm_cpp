#include "LinearAlgebra/sym_sparse_matrix.hpp"
#include <cassert>
#include <stdexcept>
#include <utility>
#include <cmath>
#include <algorithm>

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

    Scalar SymSparseMatrix::norm() const {
        const Int n = size();
        std::vector<Scalar> abs_sum(n, 0.0);   // per-row-and-column absolute sum

        const auto& Ap = upper_.col_ptr();
        const auto& Ai = upper_.row_idx();
        const auto& Ax = upper_.values();

        for (Int j = 0; j < n; ++j) {
            for (Int k = Ap[j]; k < Ap[j + 1]; ++k) {
                const Int    i = Ai[k];
                const Scalar a = std::abs(Ax[k]);
                abs_sum[i] += a;              // contributes to line i
                if (i != j) abs_sum[j] += a;  // mirror: also to line j
            }
        }

        Scalar best = 0.0;
        for (Int i = 0; i < n; ++i) best = std::max(best, abs_sum[i]);
        return best;
    }


}  // namespace ippmm