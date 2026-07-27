#include "doctest.h"
#include <limits>
#include "IPM_Control/ratio_test.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("ratio_test: two-sided primal and dual, tau backoff") {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 0, 0}, {}, {});
    std::vector<double> c(2, 0.0), b{0.0};
    std::vector<double> lb{0.0, -INF};   // x0 boxed, x1 upper-only
    std::vector<double> ub{5.0,  3.0};
    QPProblem qp{c, A, Q, b, lb, ub};

    // x0 at 2 (box [0,5]); x1 at 1 (upper 3).
    const std::vector<double> x  { 2.0, 1.0};
    const std::vector<double> dx {-1.0, 4.0};   // x0 decreasing, x1 increasing
    // Primal caps: x0 lower (2-0... i.e. (lb-x)/dx = (0-2)/(-1) = 2);
    //              x1 upper ((ub-x)/dx = (3-1)/4 = 0.5)  -> min = 0.5
    const std::vector<double> z_l { 4.0, 0.0}, dz_l {-8.0, 0.0}; // x0: -4/-8 = 0.5
    const std::vector<double> z_u { 1.0, 8.0}, dz_u { 0.0,-1.0}; // x1: -8/-1 = 8
    // Dual caps: x0 z_l -> 2 ; x1 z_u -> 8  -> min = 2

    const StepLengths s = ratio_test(qp, x, dx, z_l, dz_l, z_u, dz_u, /*tau=*/1.0);
    CHECK(s.alpha_x == doctest::Approx(0.5));   // upper of x1 binds
    CHECK(s.alpha_z  == doctest::Approx(0.5));   // z_l of x0 binds

    // With the real tau = 0.995, both scale.
    const StepLengths st = ratio_test(qp, x, dx, z_l, dz_l, z_u, dz_u);  // default tau
    CHECK(st.alpha_x == doctest::Approx(0.995 * 0.5));
    CHECK(st.alpha_z == doctest::Approx(0.995 * 0.5));
}

TEST_CASE("ratio_test: unconstrained direction gives full step") {
    SparseMatrix    A(1, 1, {0, 1}, {0}, {1.0});
    SymSparseMatrix Q(1, {0, 0}, {}, {});
    std::vector<double> c(1, 0.0), b{0.0};
    std::vector<double> lb{0.0}, ub{INF};   // x0 lower-only
    QPProblem qp{c, A, Q, b, lb, ub};

    // Moving x0 UP and z_l UP: nothing decreasing, so no cap -> alpha = tau*1.
    const StepLengths s = ratio_test(qp, {5.0}, {2.0}, {3.0}, {1.0}, {0.0}, {0.0});
    CHECK(s.alpha_x == doctest::Approx(0.995));
    CHECK(s.alpha_z == doctest::Approx(0.995));
}