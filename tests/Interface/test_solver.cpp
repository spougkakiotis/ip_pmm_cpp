#include "doctest.h"
#include <limits>
#include <cmath>
#include "Interface/solver.hpp"
#include "IPM_Control/residuals.hpp"
#include "LinearAlgebra/vector_ops.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("solve: simple equality-constrained QP") {
    // min 1/2(x0^2 + x1^2)  s.t.  x0 + x1 = 2,  x >= 0.
    // Analytic solution: x0 = x1 = 1 (symmetry), y = -1 ... check via KKT/residuals.
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});   // [1 1]
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {1.0, 1.0});      // I
    std::vector<double> c{0.0, 0.0}, b{2.0};
    std::vector<double> lb{0.0, 0.0}, ub{INF, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    SolveOptions opts; opts.tol = 1e-8; opts.maxit = 100;
    const SolveResult r = solve(qp, opts);

    REQUIRE(r.status == SolveStatus::Optimal);
    CHECK(r.x[0] == doctest::Approx(1.0).epsilon(1e-5));
    CHECK(r.x[1] == doctest::Approx(1.0).epsilon(1e-5));

    // Verify KKT: primal feasibility A x = b, and low residuals.
    const std::vector<double> rp = primal_residual(qp, r.x);
    CHECK(norm(rp, Norm::Inf) < 1e-6);
    const std::vector<double> rd = dual_residual(qp, r.x, r.y, r.z_l, r.z_u);
    CHECK(norm(rd, Norm::Inf) < 1e-6);
}

TEST_CASE("solve: box-constrained QP where bound is active") {
    // min 1/2(x0-5)^2 + 1/2(x1-5)^2  s.t. x0 + x1 = 2, 0 <= x <= 3.
    // Unconstrained min wants x=5,5 but sum=2 forces them down; symmetric -> x0=x1=1.
    // Q = I; c = [-5,-5] (from expanding (x-5)^2); constant dropped.
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {1.0, 1.0});
    std::vector<double> c{-5.0, -5.0}, b{2.0};
    std::vector<double> lb{0.0, 0.0}, ub{3.0, 3.0};
    QPProblem qp{c, A, Q, b, lb, ub};

    SolveOptions opts; opts.tol = 1e-8;
    const SolveResult r = solve(qp, opts);

    REQUIRE(r.status == SolveStatus::Optimal);
    CHECK(r.x[0] == doctest::Approx(1.0).epsilon(1e-4));
    CHECK(r.x[1] == doctest::Approx(1.0).epsilon(1e-4));
}