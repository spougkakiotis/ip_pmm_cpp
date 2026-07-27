#pragma once

#include <vector>
#include "qp_problem.hpp"
#include "KKT/kkt_solver.hpp"
#include "KKT/newton_backsolve.hpp"

namespace ippmm {

    // Mehrotra predictor-corrector search direction for two-sided IP-PMM.
    // solver must already be factorized on K for the current point.
    //   res_p, res_d : the REGULARIZED residuals (from reg_primal_res/reg_dual_res)
    //   mu           : current duality measure
    //   n_bounds     : n_l + n_u (finite-bound count), for the cubic sigma
    // Returns the combined (predictor + corrector) direction.
    NewtonDirection predictor_corrector(const KKTSolver& solver,
                                        const QPProblem& qp,
                                        const std::vector<Scalar>& x,
                                        const std::vector<Scalar>& z_l,
                                        const std::vector<Scalar>& z_u,
                                        const std::vector<Scalar>& res_p,
                                        const std::vector<Scalar>& res_d,
                                        Scalar mu,
                                        Int n_bounds,
                                        Scalar tau = 0.995);

}  