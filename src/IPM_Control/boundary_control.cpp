#include "IPM_Control/boundary_control.hpp"

#include <cmath>
#include <limits>

namespace ippmm {

Scalar boundary_control(const QPProblem& qp,
                        std::vector<Scalar>& x,
                        std::vector<Scalar>& z_l,
                        std::vector<Scalar>& z_u,
                        Scalar mu_prev) {
    const Int n = qp.num_vars();
    const Scalar eps10 = 10.0 * std::numeric_limits<Scalar>::epsilon();
    const Scalar nudge = 1e-11;

    bool activate = false;

    // Find current minima of the slacks/duals over finite sides.
    Scalar min_gl = std::numeric_limits<Scalar>::infinity();
    Scalar min_zl = std::numeric_limits<Scalar>::infinity();
    Scalar min_gu = std::numeric_limits<Scalar>::infinity();
    Scalar min_zu = std::numeric_limits<Scalar>::infinity();
    for (Int i = 0; i < n; ++i) {
        if (qp.has_lower(i)) { min_gl = std::min(min_gl, x[i] - qp.lb[i]); min_zl = std::min(min_zl, z_l[i]); }
        if (qp.has_upper(i)) { min_gu = std::min(min_gu, qp.ub[i] - x[i]); min_zu = std::min(min_zu, z_u[i]); }
    }

    // Lower side: nudge x up / z_l up if too close.
    if (min_gl < eps10) { for (Int i = 0; i < n; ++i) if (qp.has_lower(i)) x[i]   += nudge; activate = true; }
    if (min_zl < eps10) { for (Int i = 0; i < n; ++i) if (qp.has_lower(i)) z_l[i] += nudge; activate = true; }
    // Upper side: nudge x down / z_u up if too close.
    if (min_gu < eps10) { for (Int i = 0; i < n; ++i) if (qp.has_upper(i)) x[i]   -= nudge; activate = true; }
    if (min_zu < eps10) { for (Int i = 0; i < n; ++i) if (qp.has_upper(i)) z_u[i] += nudge; activate = true; }

    if (!activate) return mu_prev;

    // Recompute mu = (g_lᵀz_l + g_uᵀz_u) / (n_l + n_u).
    Scalar comp = 0.0; Int nb = 0;
    for (Int i = 0; i < n; ++i) {
        if (qp.has_lower(i)) { comp += (x[i] - qp.lb[i]) * z_l[i]; ++nb; }
        if (qp.has_upper(i)) { comp += (qp.ub[i] - x[i]) * z_u[i]; ++nb; }
    }
    return (nb > 0) ? comp / static_cast<Scalar>(nb) : mu_prev;
}

}  // namespace ippmm