#pragma once

#include <vector>
#include "LinearAlgebra/sparse_matrix.hpp"

namespace ippmm{

/* RAII wrapper around QDLDL for a symmetric quasidefinite system K x = b.
   Three-tier lifetime:
     construct  -> symbolic analysis (elimination tree), ONCE per pattern
     factorize  -> numeric factorization, once per set of values (per IPM iter)
     solve      -> triangular solves, any number of times per factorization
  
   K is supplied in CSC form as its UPPER TRIANGLE only (QDLDL's requirement).
   The sparsity pattern is fixed at construction; factorize() accepts new
   values on that same pattern. */   
    class KKTSolver{
        public:
            // Symbolic analysis. col_ptr/row_idx describe the upper-triangular
            // pattern of the n x n system. Throws std::invalid_argument if the
            // pattern is structurally unsuitable for an LDL factorization.
            KKTSolver(Int n,
                      std::vector<Int> col_ptr,
                      std::vector<Int> row_idx);

            // Numeric factorization on the fixed pattern. `values` must be the
            // upper-triangular data matching the pattern given at construction.
            // Returns false if the matrix is not factorizable (a zero pivot) —
            // the caller treats this as a numerical failure.
            bool factorize(const std::vector<Scalar>& values);

            // Solve K x = rhs, returning x. Requires a prior successful factorize().
            // May be called repeatedly against one factorization (predictor + corrector).
            std::vector<Scalar> solve(const std::vector<Scalar>& rhs) const;

            Int size() const { return n_; }
        private:
            Int n_ = 0;
            // Fixed pattern of the upper triangle of K (owned copy).
            std::vector<Int> Ap_;   // col_ptr, size n+1
            std::vector<Int> Ai_;   // row_idx, size nnz

            // Symbolic results (computed once in the constructor).
            std::vector<Int> etree_;  // size n
            std::vector<Int> Lnz_;    // size n
            Int sum_Lnz_ = 0;         // total nonzeros in L

            // Numeric factors (filled by factorize()).
            std::vector<Int>    Lp_;    // size n+1
            std::vector<Int>    Li_;    // size sum_Lnz_
            std::vector<Scalar> Lx_;    // size sum_Lnz_
            std::vector<Scalar> D_;     // size n
            std::vector<Scalar> Dinv_;  // size n

            // Scratch buffers, sized once, reused every factorize().
            std::vector<unsigned char> bwork_;  // size n   (QDLDL_bool)
            std::vector<Int>           iwork_;  // size 3n
            std::vector<Scalar>        fwork_;  // size n

            bool factorized_ = false;

    };

}