#pragma once

#include "IPM_Control/solver_state.hpp"
#include "qp_problem.hpp"
#include "KKT/kkt_system.hpp"
#include "KKT/kkt_solver.hpp"

namespace ippmm {

enum class IterStatus { Ok, Instability };

// Advance the iterate one IP-PMM step. Borrows the prebuilt KKT machinery
// (fixed pattern + symbolic factorization), reassembling values and refactoring
// each call. Mutates s (iterate, mu, mu_rate, residual norms). Parameter/estimate
// updates and stopping are handled by the outer loop.
IterStatus ip_pmm_iteration(SolverState& s,
                            const QPProblem& qp,
                            KKTSystem& system,
                            KKTSolver& solver);

}  // namespace ippmm