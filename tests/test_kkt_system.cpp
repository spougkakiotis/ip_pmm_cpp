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