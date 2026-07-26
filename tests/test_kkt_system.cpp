#include "doctest.h"
#include "kkt_system.hpp"

using namespace ippmm;

// n=2, m=1 hand example.
// Q (upper) = [2 1 ; . 5];  A = [3 4].
static QPProblem make_kkt_qp() {
    SymSparseMatrix Q(2, {0, 1, 3}, {0, 0, 1}, {2.0, 1.0, 5.0}); // col0:(0,2) col1:(0,1),(1,5)
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {3.0, 4.0});      // [3 4]
    std::vector<double> c{0.0, 0.0};
    std::vector<double> b{0.0};
    std::vector<char>   is_free{Bounded, Bounded};
    return QPProblem{c, A, Q, b, is_free};
}

TEST_CASE("KKTSystem builds the correct fixed pattern") {
    QPProblem qp = make_kkt_qp();
    KKTSystem K(qp);

    CHECK(K.size() == 3);

    // Expected upper-triangular pattern (from the hand walk):
    //   col 0 (x0): (0,0)                  -> rows {0}
    //   col 1 (x1): (0,1),(1,1)            -> rows {0,1}
    //   col 2 (y0): (0,2),(1,2),(2,2)      -> rows {0,1,2}
    const std::vector<int> expected_col_ptr{0, 1, 3, 6};
    const std::vector<int> expected_row_idx{0, 0, 1, 0, 1, 2};

    CHECK(K.col_ptr() == expected_col_ptr);
    CHECK(K.row_idx() == expected_row_idx);
}

TEST_CASE("KKTSystem assembles correct values") {
    QPProblem qp = make_kkt_qp();     // Q=[2 1;. 5], A=[3 4]
    KKTSystem K(qp);

    // Θ⁻¹ = [10, 20], ρ = 1, δ = 7.
    const std::vector<double> theta_inv{10.0, 20.0};
    const SymSparseMatrix Kmat = K.assemble(theta_inv, /*rho=*/1.0, /*delta=*/7.0);

    // Expected upper-triangular values, column by column:
    //   col 0 (x0): (0,0) = -(2 + 10 + 1) = -13
    //   col 1 (x1): (0,1) = -1 ; (1,1) = -(5 + 20 + 1) = -26
    //   col 2 (y0): (0,2) = 3 ; (1,2) = 4 ; (2,2) = 7
    const std::vector<double>& v = Kmat.values();
    REQUIRE(v.size() == 6);
    CHECK(v[0] == doctest::Approx(-13.0));
    CHECK(v[1] == doctest::Approx(-1.0));
    CHECK(v[2] == doctest::Approx(-26.0));
    CHECK(v[3] == doctest::Approx(3.0));
    CHECK(v[4] == doctest::Approx(4.0));
    CHECK(v[5] == doctest::Approx(7.0));
}

// Same Q and A as make_kkt_qp(), but x1 is now FREE instead of Bounded.
//   Q (upper) = [2 1 ; . 5];  A = [3 4]
static QPProblem make_kkt_qp_free_x1() {
    SymSparseMatrix Q(2, {0, 1, 3}, {0, 0, 1}, {2.0, 1.0, 5.0});
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {3.0, 4.0});
    std::vector<double> c{0.0, 0.0};
    std::vector<double> b{0.0};
    std::vector<char>   is_free{Bounded, Free};   // x1 free
    return QPProblem{c, A, Q, b, is_free};
}

TEST_CASE("KKTSystem pattern is unaffected by free variables") {
    // Invariant: free/bounded lives in the caller's Θ⁻¹, NOT in the pattern.
    // A free variable must still get its reserved diagonal (ρ lands there).
    KKTSystem K(make_kkt_qp_free_x1());  // note: temporary QPProblem lifetime — see comment below

    CHECK(K.size() == 3);
    CHECK(K.col_ptr() == std::vector<int>{0, 1, 3, 6});
    CHECK(K.row_idx() == std::vector<int>{0, 0, 1, 0, 1, 2});
}

TEST_CASE("KKTSystem assembly with a free variable (Θ⁻¹ zeroed at free index)") {
    QPProblem qp = make_kkt_qp_free_x1();
    KKTSystem K(qp);

    // Caller's responsibility: Θ⁻¹ is 0 at the free index (x1).
    // This test assumes that has been done; it does NOT verify the zeroing itself.
    const std::vector<double> theta_inv{10.0, 0.0};   // x0 bounded, x1 free
    const SymSparseMatrix Kmat = K.assemble(theta_inv, /*rho=*/1.0, /*delta=*/7.0);

    // Expected values (only the x1 diagonal differs from the all-bounded case):
    //   col 0 (x0): (0,0) = -(2 + 10 + 1) = -13
    //   col 1 (x1): (0,1) = -1 ; (1,1) = -(5 + 0 + 1) = -6   <-- free: θ⁻¹ = 0
    //   col 2 (y0): (0,2) = 3 ; (1,2) = 4 ; (2,2) = 7
    const std::vector<double>& v = Kmat.values();
    REQUIRE(v.size() == 6);
    CHECK(v[0] == doctest::Approx(-13.0));
    CHECK(v[1] == doctest::Approx(-1.0));
    CHECK(v[2] == doctest::Approx(-6.0));    // the one value that moved
    CHECK(v[3] == doctest::Approx(3.0));
    CHECK(v[4] == doctest::Approx(4.0));
    CHECK(v[5] == doctest::Approx(7.0));
}