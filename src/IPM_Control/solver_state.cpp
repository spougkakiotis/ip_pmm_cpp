#include "IPM_Control/solver_state.hpp"

#include "IPM_Control/warm_start.hpp"
#include "IPM_Control/complementarity.hpp"
#include "IPM_Control/reg_limit.hpp"

namespace ippmm {

SolverState initialize_state(const QPProblem& qp, Scalar tol) {
    // Rough warm-start point, then made strictly interior.
    WarmStartPoint p = warm_start_point(qp, /*delta0=*/1e-3);
    finalize_start(qp, p, /*kappa=*/0.25);

    SolverState s;
    s.x   = std::move(p.x);
    s.y   = std::move(p.y);
    s.z_l = std::move(p.z_l);
    s.z_u = std::move(p.z_u);

    // PMM centers start at the initial point:  zeta = x, lambda = y.
    s.zeta   = s.x;
    s.lambda = s.y;

    // Initial regularization (your driver: rho = delta = 8) and floor.
    s.rho       = 8.0;
    s.delta     = 8.0;
    s.reg_limit = reg_limit(qp, tol);

    // Initial mu from complementarity (0 in pure-PMM mode).
    const Complementarity c = compute_complementarity(qp, s.x, s.z_l, s.z_u);
    s.mu = c.mu;
    // mu_rate is left at 0; first set at the end of iteration 1 (per the driver).

    return s;
}

}  // namespace ippmm