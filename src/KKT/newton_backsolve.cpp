#include "KKT/newton_backsolve.hpp"

#include <cassert>

namespace ippmm {

    NewtonDirection newton_backsolve(const KKTSolver& solver,
                                     const QPProblem& qp,
                                     const std::vector<Scalar>& x,
                                     const std::vector<Scalar>& z_l,
                                     const std::vector<Scalar>& z_u,
                                     const std::vector<Scalar>& res_p,
                                     const std::vector<Scalar>& res_d,
                                     const std::vector<Scalar>& res_mu_l,
                                     const std::vector<Scalar>& res_mu_u) {
        const Int n = qp.num_vars();
        const Int m = qp.num_constraints();
        assert(static_cast<Int>(res_d.size()) == n);
        assert(static_cast<Int>(res_p.size()) == m);
        assert(solver.size() == n + m);

        // --- Build the reduced rhs = [ res_d - res_mu_l/g_l + res_mu_u/g_u ; res_p ] ---
        std::vector<Scalar> rhs(n + m, 0.0);
        for (Int i = 0; i < n; ++i) {
            Scalar xi = res_d[i];
            if (qp.has_lower(i)) {
                const Scalar g_l = x[i] - qp.lb[i];
                xi -= res_mu_l[i] / g_l;
            }
            if (qp.has_upper(i)) {
                const Scalar g_u = qp.ub[i] - x[i];
                xi += res_mu_u[i] / g_u;
            }
            rhs[i] = xi;
        }
        for (Int k = 0; k < m; ++k) rhs[n + k] = res_p[k];

        // --- Solve K [dx; dy] = rhs (single QDLDL solve; IR deferred) ---
        const std::vector<Scalar> lhs = solver.solve(rhs);

        NewtonDirection d;
        d.dx.assign(lhs.begin(),         lhs.begin() + n);
        d.dy.assign(lhs.begin() + n,     lhs.end());
        d.dz_l.assign(n, 0.0);
        d.dz_u.assign(n, 0.0);

        // --- Recover bound-dual directions ---
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) {
                const Scalar g_l = x[i] - qp.lb[i];
                d.dz_l[i] = (res_mu_l[i] - z_l[i] * d.dx[i]) / g_l;
            }
            if (qp.has_upper(i)) {
                const Scalar g_u = qp.ub[i] - x[i];
                d.dz_u[i] = (res_mu_u[i] + z_u[i] * d.dx[i]) / g_u;
            }
        }
        return d;
    }

}  // namespace ippmm