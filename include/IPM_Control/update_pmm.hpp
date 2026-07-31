#pragma once

#include <vector>
#include "IPM_Control/solver_state.hpp"

namespace ippmm {

    // Update PMM estimates and regularization from residual progress (port of
    // update_PMM_parameters). If either residual improved by >= 5%, accept the new
    // estimates (zeta<-x, lambda<-y) and shrink rho/delta at mu_rate; otherwise keep
    // estimates, shrink slower (0.666*mu_rate), and bump the stall counters.
    // Uses 2-norms of the non-regularized residuals. Mutates s.
    void update_pmm_parameters(SolverState& s,
                            const std::vector<Scalar>& nr_res_p,
                            const std::vector<Scalar>& new_nr_res_p,
                            const std::vector<Scalar>& nr_res_d,
                            const std::vector<Scalar>& new_nr_res_d);

}  // namespace ippmm