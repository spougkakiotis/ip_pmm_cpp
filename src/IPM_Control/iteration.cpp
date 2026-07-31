#include "IPM_Control/iteration.hpp"

#include <algorithm>
#include <cmath>

#include "IPM_Control/boundary_control.hpp"
#include "IPM_Control/complementarity.hpp"
#include "IPM_Control/regularized_residuals.hpp"
#include "IPM_Control/ratio_test.hpp"
#include "IPM_Control/predictor_corrector.hpp"
#include "IPM_Control/residuals.hpp"     // primal_residual, dual_residual
#include "LinearAlgebra/vector_ops.hpp"  // norm, all_finite, axpy

namespace ippmm {

    IterStatus ip_pmm_iteration(SolverState& s,
                                const QPProblem& qp,
                                KKTSystem& system,
                                KKTSolver& solver,
                                const std::vector<Scalar>& res_p,
                                const std::vector<Scalar>& res_d) {
        // Boundary control (only with inequalities). May refine mu.
        if (qp.num_bounds() > 0) {
            s.mu = boundary_control(qp, s.x, s.z_l, s.z_u, s.mu);
        }

        // Complementarity at the (possibly nudged) point -> theta_inv, mu, n_bounds.
        const Complementarity comp = compute_complementarity(qp, s.x, s.z_l, s.z_u);

        // Assemble K on the fixed pattern and factor. Bail on failure.
        const SymSparseMatrix K = system.assemble(comp.theta_inv, s.rho, s.delta);
        if (!solver.factorize(K.values())) return IterStatus::Instability;

        // Predictor-corrector search direction.
        const NewtonDirection d = predictor_corrector(
            solver, qp, s.x, s.z_l, s.z_u, res_p, res_d, comp.mu, comp.n_bounds);
        if (!all_finite(d.dx) || !all_finite(d.dy) ||
            !all_finite(d.dz_l) || !all_finite(d.dz_u)) {
            return IterStatus::Instability;
        }

        // Step lengths. With bounds: fraction-to-boundary on the combined
        // direction. Without bounds (pure PMM): Newton is exact -> full step.
        StepLengths a;
        if (comp.n_bounds > 0) {
            a = ratio_test(qp, s.x, d.dx, s.z_l, d.dz_l, s.z_u, d.dz_u);
        } else {
            a.alpha_x = 1.0;
            a.alpha_z = 1.0;
        }

        // Take the step:  x with alpha_x; y, z_l, z_u with alpha_z.
        axpy(a.alpha_x, d.dx,   s.x);
        axpy(a.alpha_z, d.dy,   s.y);
        axpy(a.alpha_z, d.dz_l, s.z_l);
        axpy(a.alpha_z, d.dz_u, s.z_u);

        // End-of-iteration mu and mu_rate from the new point.
        if (comp.n_bounds > 0) {
            const Scalar mu_prev = s.mu;
            const Complementarity cnew = compute_complementarity(qp, s.x, s.z_l, s.z_u);
            s.mu = cnew.mu;
            Scalar rate = std::abs((s.mu - mu_prev) / std::max(s.mu, mu_prev));
            rate = std::min(rate, Scalar(0.9));
            rate = std::max(rate, Scalar(0.2));
            s.mu_rate = rate;
        } else {
            s.mu_rate = 0.9;
        }

        return IterStatus::Ok;
    }

}  // namespace ippmm