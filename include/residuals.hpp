#pragma once

#include <vector>
#include "qp_problem.hpp"

namespace ippmm{

    // Primal infeasibility: r_p = Ax-b
    std::vector<Scalar> primal_residual(const QPProblem& qp,
                                        const std::vector<Scalar>& x);
    // Dual infeasibility: r_d = c+Qx-A^t y- z
    std::vector<Scalar> dual_residual(const QPProblem& qp,
                                      const std::vector<Scalar>& x,
                                      const std::vector<Scalar>& y,
                                      const std::vector<Scalar>& z);
}