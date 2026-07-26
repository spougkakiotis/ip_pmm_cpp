#include "doctest.h"
#include "qp_problem.hpp"
#include "residuals.hpp"
#include <stdexcept>
#include "sym_sparse_matrix.hpp"
#include <limits>
static constexpr double INF = std::numeric_limits<double>::infinity();

using namespace ippmm;

static QPProblem make_test_qp() {
    // A = [1 0 2 ; 0 3 1]                        (2 × 3)
    SparseMatrix A(2, 3, {0, 1, 2, 4}, {0, 1, 0, 1}, {1.0, 3.0, 2.0, 1.0});
    // Q = [2 0 1; 0 4 0; 1 0 3], upper-triangular:
    //   col0:(0,2)  col1:(1,4)  col2:(0,1),(2,3)
    SymSparseMatrix Q(3, {0, 1, 2, 4}, {0, 1, 0, 2}, {2.0, 4.0, 1.0, 3.0});
    std::vector<double> c{1.0, 1.0, 1.0};
    std::vector<double> b{5.0, 4.0};
    std::vector<double> lb{0.0, 0.0, -INF};   // x0,x1 bounded below; x2 free
    std::vector<double> ub{INF, INF,  INF};
    return QPProblem{c, A, Q, b, lb, ub};   
}

TEST_CASE("QPProblem accessors and validation") {
    QPProblem qp = make_test_qp();
    CHECK(qp.num_vars() == 3);
    CHECK(qp.num_constraints() == 2);
    CHECK(qp.is_free(2) == true);        // x2: lb=-inf, ub=+inf
    CHECK(qp.is_free(0) == false);       // x0: lb=0
    CHECK(qp.has_lower(0) == true);
    CHECK(qp.has_upper(0) == false);     // ub=+inf
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

TEST_CASE("dual residual r_d = c + Q x - Aᵀ y - z_l + z_u") {
    QPProblem qp = make_test_qp();
    // Q x = [5,8,10];  c = [1,1,1];  Aᵀy = [1,6,4]  (y = [1,2])
    // z_l = [1,1,1],  z_u = [0,2,0]
    // r_d = c + Qx - Aᵀy - z_l + z_u
    //     = [1+5-1-1+0, 1+8-6-1+2, 1+10-4-1+0] = [4, 4, 6]
    const std::vector<double> rd =
        dual_residual(qp, {1.0, 2.0, 3.0}, {1.0, 2.0},
                      /*z_l=*/{1.0, 1.0, 1.0}, /*z_u=*/{0.0, 2.0, 0.0});
    REQUIRE(rd.size() == 3);
    CHECK(rd[0] == doctest::Approx(4.0));
    CHECK(rd[1] == doctest::Approx(4.0));   // the +z_u=2 shifts this from 2 to 4
    CHECK(rd[2] == doctest::Approx(6.0));
}

TEST_CASE("validate rejects crossed bounds") {
    QPProblem qp = make_test_qp();
    qp.lb[0] = 5.0;
    qp.ub[0] = 2.0;                        // lb > ub
    CHECK_THROWS_AS(qp.validate(), std::invalid_argument);
}

TEST_CASE("bound predicates classify all four regimes") {
    // Only lb/ub matter here; A and Q are minimal but valid.
    SparseMatrix    A(1, 4, {0, 1, 1, 1, 1}, {0}, {1.0});   // 1×4, one nonzero
    SymSparseMatrix Q(4, {0, 1, 1, 1, 1}, {0}, {1.0});       // 4×4, one diagonal nonzero
    std::vector<double> c(4, 0.0), b{0.0};
    std::vector<double> lb{0.0,  0.0, -INF, -INF};
    std::vector<double> ub{1.0,  INF,  1.0,  INF};
    QPProblem qp{c, A, Q, b, lb, ub};

    CHECK((qp.has_lower(0) && qp.has_upper(0)));   // x0 boxed
    CHECK((qp.has_lower(1) && !qp.has_upper(1)));  // x1 lower-only
    CHECK((!qp.has_lower(2) && qp.has_upper(2)));  // x2 upper-only
    CHECK(qp.is_free(3));                          // x3 free
}