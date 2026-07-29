#include "doctest.h"
#include "IPM_Control/solver_state.hpp"
#include "IPM_Control/complementarity.hpp"   // add includes at top
#include "qp_problem.hpp"
#include <limits>
static constexpr double INF = std::numeric_limits<double>::infinity();

using namespace ippmm;

TEST_CASE("SolverState default-constructs to a clean zero state") {
    SolverState s;
    CHECK(s.rho == 0.0);
    CHECK(s.delta == 0.0);
    CHECK(s.mu == 0.0);
    CHECK(s.no_primal_update == 0);
    CHECK(s.x.empty());
    CHECK(s.zeta.empty());
}

TEST_CASE("initialize_state produces a consistent interior start") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{1.0, 1.0}, b{4.0};
    std::vector<double> lb{0.0, 0.0}, ub{10.0, INF};   // x0 boxed, x1 lower-only
    QPProblem qp{c, A, Q, b, lb, ub};

    const SolverState s = initialize_state(qp, /*tol=*/1e-6);

    CHECK(s.rho == doctest::Approx(8.0));
    CHECK(s.delta == doctest::Approx(8.0));
    CHECK(s.reg_limit > 0.0);
    CHECK(s.mu > 0.0);                     // bounds present -> mu > 0
    CHECK(s.zeta == s.x);                  // zeta = x
    CHECK(s.lambda == s.y);                // lambda = y
    // Strictly interior on bounded sides.
    CHECK(s.x[0] > 0.0);  CHECK(s.x[0] < 10.0);  // boxed strictly inside
    CHECK(s.x[1] > 0.0);                          // lower-only slack > 0
    CHECK(s.z_l[0] > 0.0);                        // active duals positive
}

TEST_CASE("initialize_state: pure-PMM (all free) gives mu = 0") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{1.0, 1.0}, b{4.0};
    std::vector<double> lb{-INF, -INF}, ub{INF, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    const SolverState s = initialize_state(qp, 1e-6);
    CHECK(s.mu == 0.0);                    // no bounds -> pure PMM
}