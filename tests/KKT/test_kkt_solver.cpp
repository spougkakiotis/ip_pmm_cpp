#include "doctest.h"
#include "KKT/kkt_solver.hpp"

using namespace ippmm;

TEST_CASE("KKTSolver factors and solves a small quasidefinite system") {
    // K = [ -2  1  0 ;  1  3  1 ;  0  1  2 ]  (symmetric, quasidefinite)
    // Upper triangle only, in CSC:
    const std::vector<int> col_ptr{0, 1, 3, 5};
    const std::vector<int> row_idx{0, 0, 1, 1, 2};
    const std::vector<double> values{-2.0, 1.0, 3.0, 1.0, 2.0};

    KKTSolver solver(3, col_ptr, row_idx);   // symbolic (once)

    REQUIRE(solver.factorize(values) == true);  // numeric

    // Chosen x = [1,2,3]  =>  b = K x = [0,10,8]
    const std::vector<double> x = solver.solve({0.0, 10.0, 8.0});

    REQUIRE(x.size() == 3);
    CHECK(x[0] == doctest::Approx(1.0));
    CHECK(x[1] == doctest::Approx(2.0));
    CHECK(x[2] == doctest::Approx(3.0));
}

TEST_CASE("KKTSolver reuses one factorization for multiple right-hand sides") {
    // Same K; this is the predictor/corrector pattern — one factorize, two solves.
    const std::vector<int> col_ptr{0, 1, 3, 5};
    const std::vector<int> row_idx{0, 0, 1, 1, 2};
    const std::vector<double> values{-2.0, 1.0, 3.0, 1.0, 2.0};

    KKTSolver solver(3, col_ptr, row_idx);
    REQUIRE(solver.factorize(values));

    // rhs = e1 = [1,0,0]; solve, then re-solve with a different rhs.
    const std::vector<double> x1 = solver.solve({1.0, 0.0, 0.0});
    const std::vector<double> x2 = solver.solve({0.0, 10.0, 8.0});

    // Verify x1 by residual: K x1 should equal [1,0,0].
    auto Kx = [](const std::vector<double>& v) {
        return std::vector<double>{
            -2*v[0] + 1*v[1] + 0*v[2],
             1*v[0] + 3*v[1] + 1*v[2],
             0*v[0] + 1*v[1] + 2*v[2]};
    };
    const std::vector<double> r1 = Kx(x1);
    CHECK(r1[0] == doctest::Approx(1.0));
    CHECK(r1[1] == doctest::Approx(0.0));
    CHECK(r1[2] == doctest::Approx(0.0));

    // Second solve still correct against the same factorization.
    CHECK(x2[0] == doctest::Approx(1.0));
    CHECK(x2[1] == doctest::Approx(2.0));
    CHECK(x2[2] == doctest::Approx(3.0));
}