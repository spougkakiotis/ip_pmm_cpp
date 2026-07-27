#include "IPM_Control/ratio_test.hpp"

#include <algorithm>
#include <cassert>

namespace ippmm {

    StepLengths ratio_test(const QPProblem& qp,
                           const std::vector<Scalar>& x,   const std::vector<Scalar>& dx,
                           const std::vector<Scalar>& z_l, const std::vector<Scalar>& dz_l,
                           const std::vector<Scalar>& z_u, const std::vector<Scalar>& dz_u,
                           Scalar tau) {
        const Int n = qp.num_vars();
        assert(static_cast<Int>(x.size())   == n && static_cast<Int>(dx.size())   == n);
        assert(static_cast<Int>(z_l.size()) == n && static_cast<Int>(dz_l.size()) == n);
        assert(static_cast<Int>(z_u.size()) == n && static_cast<Int>(dz_u.size()) == n);

        Scalar alpha_x = 1.0;   // primal
        Scalar alpha_z = 1.0;   // dual

        for (Int i = 0; i < n; ++i) {
            // Primal: lower bound active and x decreasing -> cap (lb - x)/dx.
            if (qp.has_lower(i) && dx[i] < 0.0)
                alpha_x = std::min(alpha_x, (qp.lb[i] - x[i]) / dx[i]);
            // Primal: upper bound active and x increasing -> cap (ub - x)/dx.
            if (qp.has_upper(i) && dx[i] > 0.0)
                alpha_x = std::min(alpha_x, (qp.ub[i] - x[i]) / dx[i]);
            // Dual: lower multiplier decreasing -> cap -z_l/dz_l.
            if (qp.has_lower(i) && dz_l[i] < 0.0)
                alpha_z = std::min(alpha_z, -z_l[i] / dz_l[i]);
            // Dual: upper multiplier decreasing -> cap -z_u/dz_u.
            if (qp.has_upper(i) && dz_u[i] < 0.0)
                alpha_z = std::min(alpha_z, -z_u[i] / dz_u[i]);
        }

        StepLengths s;
        s.alpha_x = tau * alpha_x;
        s.alpha_z = tau * alpha_z;
        return s;
    }

}  