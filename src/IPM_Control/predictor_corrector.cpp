#include "IPM_Control/predictor_corrector.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include "IPM_Control/ratio_test.hpp"

namespace ippmm {

    NewtonDirection predictor_corrector(const KKTSolver& solver,
                                        const QPProblem& qp,
                                        const std::vector<Scalar>& x,
                                        const std::vector<Scalar>& z_l,
                                        const std::vector<Scalar>& z_u,
                                        const std::vector<Scalar>& res_p,
                                        const std::vector<Scalar>& res_d,
                                        Scalar mu,
                                        Int n_bounds,
                                        Scalar tau) {
        const Int n = qp.num_vars();
        const Int m = qp.num_constraints();

        // Slacks (per finite side; 0 elsewhere).
        std::vector<Scalar> g_l(n, 0.0), g_u(n, 0.0);
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) g_l[i] = x[i] - qp.lb[i];
            if (qp.has_upper(i)) g_u[i] = qp.ub[i] - x[i];
        }

        // ---- Predictor (affine) RHS:  res_mu = -g ∘ z ---------------------
        std::vector<Scalar> res_mu_l(n, 0.0), res_mu_u(n, 0.0);
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) res_mu_l[i] = -g_l[i] * z_l[i];   // (lb - x)*z_l
            if (qp.has_upper(i)) res_mu_u[i] = -g_u[i] * z_u[i];   // (x - ub)*z_u
        }
        const NewtonDirection aff =
            newton_backsolve(solver, qp, x, z_l, z_u, res_p, res_d, res_mu_l, res_mu_u);

        // No finite bounds => no complementarity, no centering. The affine (Newton/PMM)
        // direction is the full step; skip the corrector (and avoid dividing by mu = 0).
        if (n_bounds == 0) {
            return aff;
        }

        // ---- Affine ratio test → (alpha_x, alpha_z) -----------------------
        const StepLengths a = ratio_test(qp, x, aff.dx, z_l, aff.dz_l, z_u, aff.dz_u, tau);

        // ---- centr_measure and cubic sigma*mu -----------------------------
        Scalar centr = 0.0;
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i))
                centr += (g_l[i] + a.alpha_x * aff.dx[i]) * (z_l[i] + a.alpha_z * aff.dz_l[i]);
            if (qp.has_upper(i))
                centr += (g_u[i] - a.alpha_x * aff.dx[i]) * (z_u[i] + a.alpha_z * aff.dz_u[i]);
        }
        // sigma*mu = (centr / ((n_l+n_u)*mu))^3 * mu
        const Scalar ratio = centr / (static_cast<Scalar>(n_bounds) * mu);
        const Scalar sigma_mu = ratio * ratio * ratio * mu;

        // ---- Corrector RHS:  sigma*mu ∓ dx∘dz  (affine dx, dz) ------------
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) res_mu_l[i] = sigma_mu - aff.dx[i] * aff.dz_l[i];
            if (qp.has_upper(i)) res_mu_u[i] = sigma_mu + aff.dx[i] * aff.dz_u[i];
        }
        const std::vector<Scalar> zero_p(m, 0.0), zero_d(n, 0.0);
        const NewtonDirection cor =
            newton_backsolve(solver, qp, x, z_l, z_u, zero_p, zero_d, res_mu_l, res_mu_u);

        // ---- Combine: direction = affine + corrector ----------------------
        NewtonDirection d;
        d.dx.resize(n);   d.dz_l.resize(n);   d.dz_u.resize(n);   d.dy.resize(m);
        for (Int i = 0; i < n; ++i) {
            d.dx[i]   = aff.dx[i]   + cor.dx[i];
            d.dz_l[i] = aff.dz_l[i] + cor.dz_l[i];
            d.dz_u[i] = aff.dz_u[i] + cor.dz_u[i];
        }
        for (Int k = 0; k < m; ++k) d.dy[k] = aff.dy[k] + cor.dy[k];
        return d;
    }

}  // namespace ippmm