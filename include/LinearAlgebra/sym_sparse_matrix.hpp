#pragma once

#include <vector>
#include "LinearAlgebra/sparse_matrix.hpp"  // composes a SparseMatrix; reuses Int, Scalar

namespace ippmm {

    // A symmetric matrix stored as its UPPER TRIANGLE only (entries with row <= col),
    // in CSC. This is exactly the form QDLDL consumes, so the type doubles as the
    // currency between assembly code and the factorizer.
    //
    // Composition, not inheritance: a symmetric-upper matrix is NOT substitutable
    // for a general SparseMatrix (its product is defined differently), so we hold a
    // SparseMatrix for storage and expose only symmetric-correct operations.
    class SymSparseMatrix {
    public:
        // Build from upper-triangular CSC. Throws if any stored entry has row > col,
        // or if the matrix isn't square.
        SymSparseMatrix(Int n,
                        std::vector<Int> col_ptr,
                        std::vector<Int> row_idx,
                        std::vector<Scalar> values);

        Int size() const { return upper_.rows(); }    // n (square)
        Int nnz()  const { return upper_.nnz(); }     // stored (upper) nonzeros

        // y = A x, treating the stored upper triangle as the full symmetric matrix.
        std::vector<Scalar> multiply(const std::vector<Scalar>& x) const;

        // Raw upper-triangular CSC — what KKTSolver/QDLDL need directly.
        const std::vector<Int>&    col_ptr() const { return upper_.col_ptr(); }
        const std::vector<Int>&    row_idx() const { return upper_.row_idx(); }
        const std::vector<Scalar>& values()  const { return upper_.values(); }

    private:
        SparseMatrix upper_;  // the upper-triangle storage
    };

}  // namespace ippmm