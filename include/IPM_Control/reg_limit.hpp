#pragma once

#include "qp_problem.hpp"

namespace ippmm {
    // Regularization floor, per the MATLAB solver:
    //   base = 0.1 * min(tol, 1e-6) / max(1, max(||A||_1 ||A||_inf, ||Q||_1 ||Q||_inf))
    //   reg_limit = max(base, hard_lim), then capped at 1e-6 (QP) or 1e-8 (LP).
    Scalar reg_limit(const QPProblem& qp, Scalar tol, Scalar hard_lim = 1e-12);
}  