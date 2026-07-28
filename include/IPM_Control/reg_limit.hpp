#pragma once

#include "qp_problem.hpp"

namespace ippmm {
    // Regularization floor (port of the MATLAB reg_limit computation):
    //   denom     = max(1, max(||A||_1 ||A||_inf, ||Q||_1 ||Q||_inf))
    //   hard_lim  = 5e-8 if Q nonzero else 5e-10
    //   base      = max(0.1 * min(tol, 1e-6) / denom, hard_lim)
    //   reg_limit = min(base, 1e-6 if Q nonzero else 1e-8)
    Scalar reg_limit(const QPProblem& qp, Scalar tol);
}  