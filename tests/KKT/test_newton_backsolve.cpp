#include "doctest.h"
#include <limits>
#include "KKT/newton_backsolve.hpp"
#include "IPM_Control/complementarity.hpp"
#include "KKT/kkt_system.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("newton_backsolve: directions satisfy the Newton equations") {
    // 2 vars (x0 boxed [0,10], x1 lower-only), 1 constraint.
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});   // [1 1]
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});      // diag(2,3)
    std::vector<double> c{1.0, 1.0}, b{1.0};
    std::vector<double> lb{0.0, 0.0}, ub{10.0, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    const std::vector<double> x  {4.0, 2.0};
    const std::vector<double> z_l{1.0, 2.0};
    const std::vector<double> z_u{0.5, 0.0};   // only x0 has upper

    // theta_inv from the current point, then assemble & factor K.
    const Complementarity comp = compute_complementarity(qp, x, z_l, z_u);
    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());
    const SymSparseMatrix K = system.assemble(comp.theta_inv, /*rho=*/0.5, /*delta=*/0.5);
    REQUIRE(solver.factorize(K.values()));

    // Arbitrary residuals (as a predictor/corrector would supply).
    const std::vector<double> res_p{0.3};
    const std::vector<double> res_d{0.1, -0.2};
    const std::vector<double> res_mu_l{0.7, 0.4};
    const std::vector<double> res_mu_u{0.6, 0.0};   // only x0

    const NewtonDirection d =
        newton_backsolve(solver, qp, x, z_l, z_u, res_p, res_d, res_mu_l, res_mu_u);

    // Complementarity-row identities (definition of the dz recovery):
    //  lower: z_l*dx + g_l*dz_l = res_mu_l   (g_l = x - lb)
    //  upper: -z_u*dx + g_u*dz_u = res_mu_u  (g_u = ub - x)
    const double g_l0 = x[0] - lb[0], g_l1 = x[1] - lb[1], g_u0 = ub[0] - x[0];
    CHECK(z_l[0]*d.dx[0] + g_l0*d.dz_l[0] == doctest::Approx(res_mu_l[0]));
    CHECK(z_l[1]*d.dx[1] + g_l1*d.dz_l[1] == doctest::Approx(res_mu_l[1]));
    CHECK(-z_u[0]*d.dx[0] + g_u0*d.dz_u[0] == doctest::Approx(res_mu_u[0]));

    // dz_u on x1 (no upper bound) must be exactly 0.
    CHECK(d.dz_u[1] == 0.0);
}