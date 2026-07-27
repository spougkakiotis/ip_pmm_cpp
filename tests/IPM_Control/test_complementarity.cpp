#include "doctest.h"
#include <limits>
#include "IPM_Control/complementarity.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("complementarity: boxed, one-sided, and free variables") {
    // Minimal valid A/Q; only bounds matter here.
    SparseMatrix    A(1, 3, {0, 1, 1, 1}, {0}, {1.0});
    SymSparseMatrix Q(3, {0, 0, 0, 0}, {}, {});
    std::vector<double> c(3, 0.0), b{0.0};
    //          x0 boxed     x1 lower-only   x2 free
    std::vector<double> lb{ 0.0,  1.0, -INF};
    std::vector<double> ub{10.0,  INF,  INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    // Current point and duals (z zeroed on absent sides).
    const std::vector<double> x  {4.0, 3.0, 5.0};
    const std::vector<double> z_l{2.0, 6.0, 0.0};   // x2 free -> 0
    const std::vector<double> z_u{1.0, 0.0, 0.0};   // only x0 has upper

    const Complementarity r = compute_complementarity(qp, x, z_l, z_u);

    // Slacks:
    //  g_l = [4-0, 3-1, 0]   = [4, 2, 0]
    //  g_u = [10-4, 0,  0]   = [6, 0, 0]
    CHECK(r.g_l[0] == doctest::Approx(4.0));
    CHECK(r.g_l[1] == doctest::Approx(2.0));
    CHECK(r.g_u[0] == doctest::Approx(6.0));

    // theta_inv:
    //  x0 (boxed): z_l/g_l + z_u/g_u = 2/4 + 1/6 = 0.5 + 0.16667 = 0.66667
    //  x1 (lower): 6/2 = 3
    //  x2 (free):  0
    CHECK(r.theta_inv[0] == doctest::Approx(0.5 + 1.0/6.0));
    CHECK(r.theta_inv[1] == doctest::Approx(3.0));
    CHECK(r.theta_inv[2] == doctest::Approx(0.0));

    // n_bounds: x0 two sides (2) + x1 one (1) + x2 none (0) = 3
    CHECK(r.n_bounds == 3);

    // comp = g_lᵀz_l + g_uᵀz_u
    //      = (4*2 + 2*6 + 0) + (6*1 + 0 + 0) = (8+12) + 6 = 26
    // mu = 26 / 3
    CHECK(r.mu == doctest::Approx(26.0 / 3.0));
}