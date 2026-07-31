#pragma once

#include "qp_problem.hpp"
#include "Interface/solver_status.hpp"

namespace ippmm {

    struct SolveOptions {
        Scalar tol    = 1e-4;   // matches the MATLAB default
        int    maxit  = 100;
    };

    // Solve the convex QP  min cᵀx + ½xᵀQx  s.t. Ax = b, lb <= x <= ub.
    // Requires a presolved problem (validate() must pass; no fixed variables).
    SolveResult solve(const QPProblem& qp, const SolveOptions& opts = {});

}  // namespace ippmm