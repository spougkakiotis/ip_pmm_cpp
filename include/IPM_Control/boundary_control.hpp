#pragma once

#include <vector>
#include "qp_problem.hpp"

namespace ippmm {

    // Nudge any near-boundary primal/dual entries strictly interior, and recompute
    // mu if any nudge fired (else keep mu_prev). Port of boundary_control.m.
    // Mutates x, z_l, z_u in place; returns the (possibly updated) mu.
    Scalar boundary_control(const QPProblem& qp,
                            std::vector<Scalar>& x,
                            std::vector<Scalar>& z_l,
                            std::vector<Scalar>& z_u,
                            Scalar mu_prev);

}  