#include "IPM_Control/update_pmm.hpp"

#include <algorithm>
#include "LinearAlgebra/vector_ops.hpp"   // norm (2-norm default)

namespace ippmm {

    void update_pmm_parameters(SolverState& s,
                            const std::vector<Scalar>& nr_res_p,
                            const std::vector<Scalar>& new_nr_res_p,
                            const std::vector<Scalar>& nr_res_d,
                            const std::vector<Scalar>& new_nr_res_d) {
        // Progress condition: either residual dropped by at least 5% (2-norms).
        const bool cond =
            (0.95 * norm(nr_res_p) > norm(new_nr_res_p)) ||
            (0.95 * norm(nr_res_d) > norm(new_nr_res_d));

        const Scalar full = 1.0 - s.mu_rate;          // aggressive shrink
        const Scalar slow = 1.0 - 0.666 * s.mu_rate;  // cautious shrink

        // Dual estimate / delta.
        if (cond) {
            s.lambda = s.y;
            s.delta = std::max(s.reg_limit, s.delta * full);
        } else {
            s.delta = std::max(s.reg_limit, s.delta * slow);
            ++s.no_dual_update;
        }

        // Primal estimate / rho (same condition in the active MATLAB code).
        if (cond) {
            s.zeta = s.x;
            s.rho = std::max(s.reg_limit, s.rho * full);
        } else {
            s.rho = std::max(s.reg_limit, s.rho * slow);
            ++s.no_primal_update;
        }
    }

}  // namespace ippmm