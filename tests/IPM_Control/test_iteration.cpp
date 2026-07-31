#include "doctest.h"
#include <limits>
#include <cmath>
#include "IPM_Control/iteration.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("ip_pmm_iteration: one bounded step stays interior and finite") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{1.0, 1.0}, b{4.0};
    std::vector<double> lb{0.0, 0.0}, ub{10.0, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    SolverState s = initialize_state(qp, 1e-6);
    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());

    const Scalar mu0 = s.mu;
    const IterStatus st = ip_pmm_iteration(s, qp, system, solver);

    REQUIRE(st == IterStatus::Ok);
    for (double v : s.x) CHECK(std::isfinite(v));
    CHECK(s.x[0] > 0.0);  CHECK(s.x[0] < 10.0);
    CHECK(s.x[1] > 0.0);
    CHECK(s.z_l[0] > 0.0);
    CHECK(s.mu > 0.0);
    CHECK(s.mu <= mu0 * 10.0);
    CHECK(s.mu_rate >= 0.2);
    CHECK(s.mu_rate <= 0.9);
}

TEST_CASE("ip_pmm_iteration: pure PMM (no bounds) takes a full step") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{1.0, 1.0}, b{4.0};
    std::vector<double> lb{-INF, -INF}, ub{INF, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    SolverState s = initialize_state(qp, 1e-6);
    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());

    const std::vector<double> x_before = s.x;
    const IterStatus st = ip_pmm_iteration(s, qp, system, solver);

    REQUIRE(st == IterStatus::Ok);
    CHECK(s.mu == 0.0);
    CHECK(s.mu_rate == doctest::Approx(0.9));
    for (double v : s.x) CHECK(std::isfinite(v));
    CHECK((s.x[0] != x_before[0] || s.x[1] != x_before[1]));  // a real step was taken
}