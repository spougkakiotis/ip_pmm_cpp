#pragma once

#include <vector>
#include "qp_problem.hpp"

namespace ippmm {
 
    struct WarmStartPoint {
        std::vector<Scalar> x, y, z_l, z_u;
    };
    // Pre-shift warm-start point (x, y, z_l, z_u), before boundary guards / Mehrotra
    // shift. Computed by a direct solve of the regularized
    // least-squares augmented system, the direct analogue of Mehrotra_Warm_Start's PCG.
    WarmStartPoint warm_start_point(const QPProblem& qp, Scalar delta0 = 1e-3);

    // Make a rough warm-start point strictly interior: push near-boundary slacks and
    // multipliers off the wall, then apply a Mehrotra-style shift. Boxed primal x is
    // kept interior via a kappa-clamp; one-sided x uses the Mehrotra primal shift;
    // all bound-active duals get the Mehrotra dual shift. Mutates p in place.
    void finalize_start(const QPProblem& qp, WarmStartPoint& p, Scalar kappa = 0.25);
}  // namespace ippmm