#include "doctest.h"
#include <limits>
#include <cmath>
#include "IPM_Control/predictor_corrector.hpp"
#include "IPM_Control/complementarity.hpp"
#include "KKT/kkt_system.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("predictor_corrector: bounded case is finite and well-formed") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{1.0, 1.0}, b{1.0};
    std::vector<double> lb{0.0, 0.0}, ub{10.0, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    const std::vector<double> x  {4.0, 2.0};
    const std::vector<double> z_l{1.0, 2.0};
    const std::vector<double> z_u{0.5, 0.0};

    const Complementarity comp = compute_complementarity(qp, x, z_l, z_u);
    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());
    const SymSparseMatrix K = system.assemble(comp.theta_inv, 0.5, 0.5);
    REQUIRE(solver.factorize(K.values()));

    const std::vector<double> res_p{0.1};
    const std::vector<double> res_d{0.2, -0.1};

    const NewtonDirection d = predictor_corrector(
        solver, qp, x, z_l, z_u, res_p, res_d, comp.mu, comp.n_bounds);

    // Well-formed: right sizes, all finite, absent upper side of x1 is zero.
    REQUIRE(d.dx.size() == 2);
    REQUIRE(d.dy.size() == 1);
    for (double v : d.dx)   CHECK(std::isfinite(v));
    for (double v : d.dz_l) CHECK(std::isfinite(v));
    for (double v : d.dz_u) CHECK(std::isfinite(v));
    CHECK(d.dz_u[1] == 0.0);   // x1 has no upper bound
}

TEST_CASE("predictor_corrector: n_bounds == 0 gives the pure PMM Newton step") {
    // All variables free -> no complementarity. The direction must equal a single
    // newton_backsolve with zero res_mu (which the corrector skip guarantees).
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{1.0, 1.0}, b{1.0};
    std::vector<double> lb{-INF, -INF}, ub{INF, INF};   // fully free
    QPProblem qp{c, A, Q, b, lb, ub};

    const std::vector<double> x{4.0, 2.0}, z_l(2, 0.0), z_u(2, 0.0);
    const Complementarity comp = compute_complementarity(qp, x, z_l, z_u);
    REQUIRE(comp.n_bounds == 0);
    REQUIRE(comp.mu == 0.0);

    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());
    const SymSparseMatrix K = system.assemble(comp.theta_inv, 0.5, 0.5);
    REQUIRE(solver.factorize(K.values()));

    const std::vector<double> res_p{0.1}, res_d{0.2, -0.1};

    // Oracle: with no bounds, res_mu is all zero, so predictor_corrector must
    // return exactly newton_backsolve(res_p, res_d, 0, 0).
    const std::vector<double> zmu(2, 0.0);
    const NewtonDirection oracle =
        newton_backsolve(solver, qp, x, z_l, z_u, res_p, res_d, zmu, zmu);

    const NewtonDirection d = predictor_corrector(
        solver, qp, x, z_l, z_u, res_p, res_d, comp.mu, comp.n_bounds);

    REQUIRE(d.dx.size() == 2);
    CHECK(d.dx[0] == doctest::Approx(oracle.dx[0]));
    CHECK(d.dx[1] == doctest::Approx(oracle.dx[1]));
    CHECK(d.dy[0] == doctest::Approx(oracle.dy[0]));
}