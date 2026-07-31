#include "doctest.h"
#include <limits>
#include "IPM_Control/stopping.hpp"

using namespace ippmm;
static constexpr double INF = std::numeric_limits<double>::infinity();

static QPProblem tiny_qp() {
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {1.0, 1.0});
    SymSparseMatrix Q(2, {0, 0, 0}, {}, {});
    std::vector<double> c{2.0, 4.0};   // ||c||_inf = 4
    std::vector<double> b{3.0};        // ||b||_inf = 3
    std::vector<double> lb{0.0, 0.0}, ub{INF, INF};
    return QPProblem{c, A, Q, b, lb, ub};
}

TEST_CASE("check_stopping: converged -> Optimal") {
    QPProblem qp = tiny_qp();
    // Tiny residuals and mu, well below tol.
    auto st = check_stopping(qp, /*nr_p=*/{1e-9}, /*nr_d=*/{1e-9, 1e-9},
                             /*mu=*/1e-9, /*tol=*/1e-6);
    REQUIRE(st.has_value());
    CHECK(*st == SolveStatus::Optimal);
}

TEST_CASE("check_stopping: primal residual too large -> keep going") {
    QPProblem qp = tiny_qp();
    // p_inf/(1+3) = 1/4 = 0.25 >> tol -> not optimal.
    auto st = check_stopping(qp, {1.0}, {1e-9, 1e-9}, 1e-9, 1e-6);
    CHECK(!st.has_value());
}

TEST_CASE("check_stopping: mu too large -> keep going") {
    QPProblem qp = tiny_qp();
    auto st = check_stopping(qp, {1e-9}, {1e-9, 1e-9}, /*mu=*/1e-3, 1e-6);
    CHECK(!st.has_value());
}

TEST_CASE("check_stopping: relative scaling by ||b||,||c||") {
    QPProblem qp = tiny_qp();  // ||b||_inf=3, ||c||_inf=4
    // p_inf = 3.9e-6: 3.9e-6/(1+3)=9.75e-7 < 1e-6 -> primal ok (absolute 3.9e-6 would fail)
    // d_inf = 4.9e-6: 4.9e-6/(1+4)=9.8e-7 < 1e-6 -> dual ok
    auto st = check_stopping(qp, {3.9e-6}, {4.9e-6, 0.0}, 1e-9, 1e-6);
    REQUIRE(st.has_value());
    CHECK(*st == SolveStatus::Optimal);
}