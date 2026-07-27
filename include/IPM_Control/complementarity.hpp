#pragma once

#include <vector>
#include "qp_problem.hpp"

namespace ippmm {

    // Per-variable results of the complementarity computation at the current point.
    struct Complementarity {
        std::vector<Scalar> g_l;        // x_i - lb_i  where finite lower bound, else 0
        std::vector<Scalar> g_u;        // ub_i - x_i  where finite upper bound, else 0
        std::vector<Scalar> theta_inv;  // z_l/g_l + z_u/g_u  (per finite side), 0 if free
        Scalar mu = 0.0;                // (g_lᵀz_l + g_uᵀz_u) / (n_l + n_u)
        Int    n_bounds = 0;            // n_l + n_u  (finite bound count)
    };

    // Compute slacks, theta_inv, and mu from the current iterate.
    // z_l[i] must be 0 where there is no lower bound; z_u[i] 0 where no upper bound.
    Complementarity compute_complementarity(const QPProblem& qp,
                                            const std::vector<Scalar>& x,
                                            const std::vector<Scalar>& z_l,
                                            const std::vector<Scalar>& z_u);

}  // namespace ippmm