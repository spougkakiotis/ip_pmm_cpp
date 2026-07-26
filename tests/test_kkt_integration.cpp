#include "doctest.h"
#include <stdexcept>

#include "qp_problem.hpp"
#include "kkt_system.hpp"
#include "kkt_solver.hpp"

using namespace ippmm;

TEST_CASE("End-to-end: assemble KKT, factor, solve") {
    // Q (upper) = [2 1 ; . 5];  A = [3 4];  both variables bounded.
    SymSparseMatrix Q(2, {0, 1, 3}, {0, 0, 1}, {2.0, 1.0, 5.0});
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {3.0, 4.0});
    QPProblem qp{ std::vector<double>{0.0, 0.0}, A, Q,
                  std::vector<double>{0.0},
                  std::vector<char>{Bounded, Bounded} };

    // Build the system (pattern once) and a solver over that fixed pattern.
    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());

    // Assemble concrete values, then factor.
    const std::vector<double> theta_inv{10.0, 20.0};
    const SymSparseMatrix K = system.assemble(theta_inv, /*rho=*/1.0, /*delta=*/7.0);
    REQUIRE(solver.factorize(K.values()) == true);

    // K (full symmetric, from the assembled upper triangle) is:
    //        x0     x1     y0
    //  x0 [ -13    -1      3 ]
    //  x1 [  -1   -26      4 ]
    //  y0 [   3     4      7 ]
    //
    // Choose x = [1, 2, 3]; compute b = K x as the oracle:
    //  b0 = -13*1 +  -1*2 + 3*3 =  -6
    //  b1 =  -1*1 + -26*2 + 4*3 = -41
    //  b2 =   3*1 +   4*2 + 7*3 =  32
    const std::vector<double> sol = solver.solve({-6.0, -41.0, 32.0});

    REQUIRE(sol.size() == 3);
    CHECK(sol[0] == doctest::Approx(1.0));
    CHECK(sol[1] == doctest::Approx(2.0));
    CHECK(sol[2] == doctest::Approx(3.0));
}

TEST_CASE("End-to-end: re-assemble and re-solve on the same pattern") {
    // Proves the symbolic-once design: build pattern once, then factor/solve
    // repeatedly with DIFFERENT values — as the IPM loop does each iteration.
    SymSparseMatrix Q(2, {0, 1, 3}, {0, 0, 1}, {2.0, 1.0, 5.0});
    SparseMatrix    A(1, 2, {0, 1, 2}, {0, 0}, {3.0, 4.0});
    QPProblem qp{ std::vector<double>{0.0, 0.0}, A, Q,
                  std::vector<double>{0.0},
                  std::vector<char>{Bounded, Bounded} };

    KKTSystem system(qp);
    KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());

    // First "iteration": Θ⁻¹ = [10,20], ρ=1, δ=7  (same K as above).
    {
        const SymSparseMatrix K = system.assemble({10.0, 20.0}, 1.0, 7.0);
        REQUIRE(solver.factorize(K.values()));
        const std::vector<double> sol = solver.solve({-6.0, -41.0, 32.0});
        CHECK(sol[0] == doctest::Approx(1.0));
        CHECK(sol[1] == doctest::Approx(2.0));
        CHECK(sol[2] == doctest::Approx(3.0));
    }

    // Second "iteration": DIFFERENT Θ⁻¹ and regularization — same pattern reused.
    // K' =  x0 [ -(2+0+2)   -1        3 ] = [ -4  -1   3 ]
    //       x1 [  -1      -(5+0+2)    4 ]   [ -1  -7   4 ]
    //       y0 [   3         4        5 ]   [  3   4   5 ]   (δ = 5)
    // x = [1,1,1] => b = [ -4-1+3, -1-7+4, 3+4+5 ] = [ -2, -4, 12 ]
    {
        const SymSparseMatrix K = system.assemble({0.0, 0.0}, /*rho=*/2.0, /*delta=*/5.0);
        REQUIRE(solver.factorize(K.values()));
        const std::vector<double> sol = solver.solve({-2.0, -4.0, 12.0});
        CHECK(sol[0] == doctest::Approx(1.0));
        CHECK(sol[1] == doctest::Approx(1.0));
        CHECK(sol[2] == doctest::Approx(1.0));
    }
}