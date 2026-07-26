#pragma once

#include <vector>
#include "sparse_matrix.hpp"
#include "sym_sparse_matrix.hpp"

namespace ippmm{

    // Per-variable kind. Unscoped + char-backed on purpose: these get stored as
    // raw bytes and used as initializers a lot, so we want the ergonomic implicit conversion.
    enum VarKind : char {Bounded = 0, Free = 1};

    // Convex QP:  min cᵀx + ½ xᵀQx   s.t.  Ax = b,  x_i ≥ 0 (Bounded) / free (Free)
    // Plain data aggregate — mirrors your MATLAB IP_PMM(c, A, Q, b, free_variables).
    struct QPProblem {
        std::vector<Scalar> c;        // objective linear term, length n
        SparseMatrix        A;        // constraints, m × n
        SymSparseMatrix        Q;        // Hessian (symmetric PSD), n × n
        std::vector<Scalar> b;        // rhs, length m
        std::vector<char>   is_free;  // length n; Free / Bounded per variable

        Int num_vars()        const { return A.cols(); }  // n
        Int num_constraints() const { return A.rows(); }  // m

        bool variable_is_free(Int i) const { return is_free[i] != Bounded; }

        void validate() const;  // throws if the pieces don't fit together
    };

}