#include "doctest.h"
#include "sym_sparse_matrix.hpp"
#include "sparse_matrix.hpp"
#include <stdexcept>

using namespace ippmm;

TEST_CASE("SymSparseMatrix: symmetric matvec, hand oracle") {
    // Symmetric [2 0 1; 0 4 0; 1 0 3], upper triangle only.
    SymSparseMatrix Q(3, {0, 1, 2, 4}, {0, 1, 0, 2}, {2.0, 4.0, 1.0, 3.0});
    const std::vector<double> y = Q.multiply({1.0, 2.0, 3.0});
    REQUIRE(y.size() == 3);
    CHECK(y[0] == doctest::Approx(5.0));   // 2*1 + 1*3
    CHECK(y[1] == doctest::Approx(8.0));   // 4*2
    CHECK(y[2] == doctest::Approx(10.0));  // 1*1 + 3*3
}

TEST_CASE("SymSparseMatrix matches full general multiply") {
    SparseMatrix Q_full(3, 3, {0, 2, 3, 5}, {0, 2, 1, 0, 2},
                        {2.0, 1.0, 4.0, 1.0, 3.0});          // both triangles
    SymSparseMatrix Q_sym(3, {0, 1, 2, 4}, {0, 1, 0, 2},
                          {2.0, 4.0, 1.0, 3.0});             // upper only

    for (const std::vector<double>& x : { std::vector<double>{1.0, 2.0, 3.0},
                                          std::vector<double>{-1.0, 0.5, 2.0},
                                          std::vector<double>{7.0, -3.0, 0.0} }) {
        const std::vector<double> yf = Q_full.multiply(x);
        const std::vector<double> ys = Q_sym.multiply(x);
        REQUIRE(ys.size() == yf.size());
        for (std::size_t i = 0; i < yf.size(); ++i)
            CHECK(ys[i] == doctest::Approx(yf[i]));
    }
}

TEST_CASE("SymSparseMatrix rejects lower-triangular storage") {
    // Entry at (2,0): row 2 > col 0 -> must throw.
    CHECK_THROWS_AS(
        SymSparseMatrix(3, {0, 2, 2, 2}, {0, 2}, {1.0, 9.0}),
        std::invalid_argument);
}