#include "IPM_Control/complementarity.hpp"

#include <cassert>

namespace ippmm {

    Complementarity compute_complementarity(const QPProblem& qp,
                                            const std::vector<Scalar>& x,
                                            const std::vector<Scalar>& z_l,
                                            const std::vector<Scalar>& z_u) {
        const Int n = qp.num_vars();
        assert(static_cast<Int>(x.size())   == n);
        assert(static_cast<Int>(z_l.size()) == n);
        assert(static_cast<Int>(z_u.size()) == n);

        Complementarity c;
        c.g_l.assign(n, 0.0);
        c.g_u.assign(n, 0.0);
        c.theta_inv.assign(n, 0.0);

        Scalar comp = 0.0;   // g_lᵀz_l + g_uᵀz_u
        Int    nb   = 0;     // n_l + n_u

        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) {
                const Scalar g = x[i] - qp.lb[i];
                c.g_l[i] = g;
                c.theta_inv[i] += z_l[i] / g;
                comp += g * z_l[i];
                ++nb;
            }
            if (qp.has_upper(i)) {
                const Scalar g = qp.ub[i] - x[i];
                c.g_u[i] = g;
                c.theta_inv[i] += z_u[i] / g;
                comp += g * z_u[i];
                ++nb;
            }
        }

        c.n_bounds = nb;
        c.mu = (nb > 0) ? comp / static_cast<Scalar>(nb) : 0.0;
        return c;
    }

}  // namespace ippmm