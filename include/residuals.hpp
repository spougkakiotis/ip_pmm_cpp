#pragma once

#include <vector>
#include "qp_problem.hpp"

namespace ippmm{

    // Primal infeasibility: r_p = Ax-b
    std::vector<Scalar> primal_residual(const QPProblem& qp,
                                        const std::vector<Scalar>& x);
    // Dual infeasibility:  r_d = c + Q x - Aᵀ y - z_l + z_u   (length n)
    //   z_l : lower-bound multipliers (>= 0; 0 where there is no lower bound)
    //   z_u : upper-bound multipliers (>= 0; 0 where there is no upper bound)
    // The caller keeps z_l/z_u zeroed at absent bounds; this function is pure arithmetic.
    std::vector<Scalar> dual_residual(const QPProblem& qp,
                                    const std::vector<Scalar>& x,
                                    const std::vector<Scalar>& y,
                                    const std::vector<Scalar>& z_l,
                                    const std::vector<Scalar>& z_u);
}