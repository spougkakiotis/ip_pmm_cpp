#pragma once

#include <vector>
#include <limits>
#include "LinearAlgebra/sparse_matrix.hpp"
#include "LinearAlgebra/sym_sparse_matrix.hpp"

namespace ippmm{

    // Convex QP:  min cᵀx + ½ xᵀQx   s.t.  Ax = b,  lb ≤ x ≤ ub
    // Bounds use ±infinity() sentinels: lb[i] = -inf means "no lower bound",
    // ub[i] = +inf means "no upper bound". A free variable has both infinite.
    struct QPProblem {
        std::vector<Scalar> c;   // objective linear term, length n
        SparseMatrix        A;   // constraints, m × n
        SymSparseMatrix     Q;   // Hessian: symmetric PSD, upper-triangular, n × n
        std::vector<Scalar> b;   // rhs, length m
        std::vector<Scalar> lb;  // lower bounds, length n (-inf = unbounded below)
        std::vector<Scalar> ub;  // upper bounds, length n (+inf = unbounded above)

        Int num_vars()        const { return A.cols(); }  // n
        Int num_constraints() const { return A.rows(); }  // m

        // Derived bound predicates — no separate stored mask (single source of truth).
        bool has_lower(Int i) const { return lb[i] > -std::numeric_limits<Scalar>::infinity(); }
        bool has_upper(Int i) const { return ub[i] <  std::numeric_limits<Scalar>::infinity(); }
        bool is_free(Int i)   const { return !has_lower(i) && !has_upper(i); }

        void validate() const;
    };
}