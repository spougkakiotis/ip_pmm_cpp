#pragma once

#include <vector>
#include "qp_problem.hpp"
#include "KKT/kkt_solver.hpp"

namespace ippmm {

    // The Newton direction for the two-sided IP-PMM system.
    struct NewtonDirection {
        std::vector<Scalar> dx;
        std::vector<Scalar> dy;
        std::vector<Scalar> dz_l;
        std::vector<Scalar> dz_u;
    };

    // Solve the reduced KKT system with an already-factorized solver, then recover
    // the bound-dual directions. Port of Newton_backsolve (iterative refinement
    // deferred). Requires solver.factorize(...) to have succeeded on the current K.
    //
    //   rhs_x = res_d - res_mu_l/g_l + res_mu_u/g_u   (per finite side)
    //   rhs_y = res_p
    //   [dx; dy] = K^{-1} [rhs_x; rhs_y]
    //   dz_l = (res_mu_l - z_l*dx) / g_l              [lower]
    //   dz_u = (res_mu_u + z_u*dx) / g_u              [upper]
    //
    // g_l = x - lb, g_u = ub - x. z_l/z_u/res_mu_* are 0 on absent sides.
    NewtonDirection newton_backsolve(const KKTSolver& solver,
                                     const QPProblem& qp,
                                     const std::vector<Scalar>& x,
                                     const std::vector<Scalar>& z_l,
                                     const std::vector<Scalar>& z_u,
                                     const std::vector<Scalar>& res_p,
                                     const std::vector<Scalar>& res_d,
                                     const std::vector<Scalar>& res_mu_l,
                                     const std::vector<Scalar>& res_mu_u);

}  