#include "doctest.h"
#include <limits>
#include "IPM_Control/warm_start.hpp"
#include "LinearAlgebra/vector_ops.hpp"
#include <cmath>

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("warm_start_point: x approximately satisfies A x = b") {
    // A = [1 1], b = [4]. Regularized least-squares -> A x ≈ b (up to δ₀).
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 0, 0}, {}, {});      // empty (LP-like)
    std::vector<double> c{1.0, 1.0}, b{4.0};
    std::vector<double> lb{0.0, -INF}, ub{INF, 3.0};  // x0 lower, x1 upper
    QPProblem qp{c, A, Q, b, lb, ub};

    const WarmStartPoint p = warm_start_point(qp, /*delta0=*/1e-3);

    REQUIRE(p.x.size() == 2);
    REQUIRE(p.y.size() == 1);
    // A x should be close to b (exactly b would need δ₀=0; with δ₀=1e-3 it's near).
    const std::vector<double> Ax = A.multiply(p.x);
    CHECK(Ax[0] == doctest::Approx(4.0).epsilon(0.01));   // within ~1%

    // z split: x0 has lower -> z_l[0] set, z_u[0] = 0; x1 has upper -> z_u[1] set, z_l[1] = 0.
    CHECK(p.z_u[0] == 0.0);
    CHECK(p.z_l[1] == 0.0);
    // All finite.
    for (double v : p.x)   CHECK(std::isfinite(v));
    for (double v : p.z_l) CHECK(std::isfinite(v));
}

TEST_CASE("warm_start_point: boxed variable splits z into +/- parts (both >= 0)") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 0, 0}, {}, {});
    std::vector<double> c{1.0, 1.0}, b{4.0};
    std::vector<double> lb{0.0, 0.0}, ub{10.0, 10.0};   // both boxed
    QPProblem qp{c, A, Q, b, lb, ub};

    const WarmStartPoint p = warm_start_point(qp, 1e-3);
    // For a boxed variable, both multipliers must be >= 0 (that was the bug before).
    for (int i = 0; i < 2; ++i) {
        CHECK(p.z_l[i] >= 0.0);
        CHECK(p.z_u[i] >= 0.0);
        // and z_l - z_u == z (dual feasibility preserved): z_l*z_u one side is 0
        CHECK((p.z_l[i] == 0.0 || p.z_u[i] == 0.0));   // max(+,-) => one side zero
    }
}

TEST_CASE("finalize_start yields a strictly interior point across all regimes") {
    // x0 free, x1 lower-only [0,inf), x2 upper-only (-inf,5], x3 boxed [2,8]
    SparseMatrix    A(1, 4, {0, 1, 1, 1, 1}, {0}, {1.0});
    SymSparseMatrix Q(4, {0, 0, 0, 0, 0}, {}, {});
    std::vector<double> c(4, 0.0), b{0.0};
    std::vector<double> lb{-INF, 0.0, -INF, 2.0};
    std::vector<double> ub{ INF, INF,  5.0, 8.0};
    QPProblem qp{c, A, Q, b, lb, ub};

    WarmStartPoint p;
    p.x   = {3.0, 0.0, 5.0, 9.0};   // x1 at lb, x2 at ub, x3 above ub -> guards fire
    p.z_l = {0.0, 0.0, 0.0, 0.0};
    p.z_u = {0.0, 0.0, 0.0, 0.0};

    finalize_start(qp, p, 0.25);

    CHECK(p.x[1] > qp.lb[1]);                 // lower-only slack > 0
    CHECK(p.x[2] < qp.ub[2]);                 // upper-only slack > 0
    CHECK((p.x[3] > qp.lb[3] && p.x[3] < qp.ub[3]));  // boxed strictly inside
    CHECK(p.z_l[1] > 0.0);
    CHECK(p.z_u[2] > 0.0);
    CHECK((p.z_l[3] > 0.0 && p.z_u[3] > 0.0)); // boxed: both duals positive
    CHECK(p.z_l[0] == 0.0);                    // free var duals untouched
    CHECK(p.z_u[0] == 0.0);
}