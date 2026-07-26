#include "doctest.h"
#include "qp_problem.hpp"
#include "residuals.hpp"
#include <stdexcept>

using namespace ippmm;

static QPProblem make_test_qp() {
    // A = [1 0 2 ; 0 3 1]                        (2 × 3)
    SparseMatrix A(2, 3, {0, 1, 2, 4}, {0, 1, 0, 1}, {1.0, 3.0, 2.0, 1.0});
    // Q = [2 0 1 ; 0 4 0 ; 1 0 3]  symmetric PSD (3 × 3)
    SparseMatrix Q(3, 3, {0, 2, 3, 5}, {0, 2, 1, 0, 2}, {2.0, 1.0, 4.0, 1.0, 3.0});
    std::vector<double> c{1.0, 1.0, 1.0};
    std::vector<double> b{5.0, 4.0};
    std::vector<char>   is_free{Bounded, Bounded, Free};
    return QPProblem{c, A, Q, b, is_free};
}

TEST_CASE("QPProblem accessors and validation") {
    QPProblem qp = make_test_qp();
    CHECK(qp.num_vars() == 3);
    CHECK(qp.num_constraints() == 2);
    CHECK(qp.variable_is_free(2) == true);
    CHECK(qp.variable_is_free(0) == false);
    CHECK_NOTHROW(qp.validate());
}

TEST_CASE("validate rejects a dimension mismatch") {
    QPProblem qp = make_test_qp();
    qp.b.push_back(9.0);                    // b now length 3, but A has 2 rows
    CHECK_THROWS_AS(qp.validate(), std::invalid_argument);
}

TEST_CASE("primal residual r_p = A x - b") {
    QPProblem qp = make_test_qp();
    // A x = [7, 9];  b = [5, 4];  r_p = [2, 5]
    const std::vector<double> rp = primal_residual(qp, {1.0, 2.0, 3.0});
    REQUIRE(rp.size() == 2);
    CHECK(rp[0] == doctest::Approx(2.0));
    CHECK(rp[1] == doctest::Approx(5.0));
}

TEST_CASE("dual residual r_d = c + Q x - Aᵀ y - z") {
    QPProblem qp = make_test_qp();
    // Q x = [5,8,10];  c = [1,1,1];  Aᵀy = [1,6,4];  z = [1,1,1]
    // r_d = [1+5-1-1, 1+8-6-1, 1+10-4-1] = [4, 2, 6]
    const std::vector<double> rd =
        dual_residual(qp, {1.0, 2.0, 3.0}, {1.0, 2.0}, {1.0, 1.0, 1.0});
    REQUIRE(rd.size() == 3);
    CHECK(rd[0] == doctest::Approx(4.0));
    CHECK(rd[1] == doctest::Approx(2.0));
    CHECK(rd[2] == doctest::Approx(6.0));
}