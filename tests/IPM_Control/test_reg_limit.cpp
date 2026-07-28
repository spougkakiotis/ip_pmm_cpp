#include "doctest.h"
#include <limits>
#include "IPM_Control/reg_limit.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

TEST_CASE("reg_limit: QP, hard_lim floor bites") {
    // A_prod=2, Q_prod=9 -> denom=9 ; 0.1*1e-6/9 ≈ 1.11e-8
    // hard_lim(QP)=5e-8 > 1.11e-8, so base floored to 5e-8 ; cap 1e-6 -> 5e-8
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 1, 2}, {0, 1}, {2.0, 3.0});
    std::vector<double> c{0.0, 0.0}, b{0.0}, lb{0.0, 0.0}, ub{INF, INF};
    QPProblem qp{c, A, Q, b, lb, ub};
    CHECK(reg_limit(qp, 1e-6) == doctest::Approx(5e-8));
}

TEST_CASE("reg_limit: LP (empty Q) uses 1e-8 cap") {
    // Empty Q -> Q_prod = 0. A = [1 1] -> A_prod = 2. denom = max(1,2)=2.
    // base = 0.1 * 1e-6 / 2 = 5e-8 ; cap = 1e-8 (Q empty) -> min(5e-8, 1e-8) = 1e-8
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 0, 0}, {}, {});   // empty
    std::vector<double> c{0.0, 0.0}, b{0.0}, lb{0.0, 0.0}, ub{INF, INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    const Scalar rl = reg_limit(qp, /*tol=*/1e-6);
    CHECK(rl == doctest::Approx(1e-8));   // cap bites
}