#pragma once
#include <vector>
#include "IPM_Control/solver_state.hpp"
#include "qp_problem.hpp"
#include "KKT/kkt_system.hpp"
#include "KKT/kkt_solver.hpp"

namespace ippmm {

enum class IterStatus { Ok, Instability };

// Advance the iterate one IP-PMM step, given the regularized residuals computed
// by the caller (from the current point, before this call). Runs boundary_control,
// assembles/factors K, computes the predictor-corrector direction, and takes the
// step. Mutates s (x, y, z_l, z_u, mu, mu_rate). Residual computation, stopping,
// and PMM-parameter updates are the loop's responsibility.
IterStatus ip_pmm_iteration(SolverState& s,
                            const QPProblem& qp,
                            KKTSystem& system,
                            KKTSolver& solver,
                            const std::vector<Scalar>& res_p,
                            const std::vector<Scalar>& res_d);

}  // namespace ippmm