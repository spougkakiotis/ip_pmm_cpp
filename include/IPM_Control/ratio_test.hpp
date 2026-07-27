#pragma once

#include <vector>
#include "qp_problem.hpp"

namespace ippmm {

    struct StepLengths { Scalar alpha_x; Scalar alpha_z; };

    // Fraction-to-boundary step lengths.
    // Primal alpha_x keeps x strictly within [lb, ub]; dual alpha_z keeps z_l, z_u
    // in the positive orthant. tau in (0,1) backs off from the exact boundary.
    //   z_l[i] must be 0 / irrelevant where there is no lower bound; likewise z_u.
    StepLengths ratio_test(const QPProblem& qp,
                           const std::vector<Scalar>& x,   const std::vector<Scalar>& dx,
                           const std::vector<Scalar>& z_l, const std::vector<Scalar>& dz_l,
                           const std::vector<Scalar>& z_u, const std::vector<Scalar>& dz_u,
                           Scalar tau = 0.995);
}  