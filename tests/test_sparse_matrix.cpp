#include "doctest.h"
#include "sparse_matrix.hpp"

TEST_CASE("CSC matrix-vector product") {
    // A = [4 0 1; 0 5 0; 2 0 3]
    ippmm::SparseMatrix A(3, 3,
        {0, 2, 3, 5},          // col_ptr
        {0, 2, 1, 0, 2},       // row_idx
        {4.0, 2.0, 5.0, 1.0, 3.0});

    CHECK(A.rows() == 3);
    CHECK(A.cols() == 3);
    CHECK(A.nnz()  == 5);

    const std::vector<double> y = A.multiply({1.0, 2.0, 3.0});

    REQUIRE(y.size() == 3);
    CHECK(y[0] == doctest::Approx(7.0));
    CHECK(y[1] == doctest::Approx(10.0));
    CHECK(y[2] == doctest::Approx(11.0));
}

TEST_CASE("CSC transpose matrix-vector product"){
   // A = [4 0 1; 0 5 0; 2 0 3]
    ippmm::SparseMatrix A(3, 3,
        {0, 2, 3, 5},          // col_ptr
        {0, 2, 1, 0, 2},       // row_idx
        {4.0, 2.0, 5.0, 1.0, 3.0});
    
    const std::vector<double> y = A.multiply_tr({1.0, 2.0, 3.0});

    REQUIRE(y.size() == 3);
    CHECK(y[0] == doctest::Approx(10.0));
    CHECK(y[1] == doctest::Approx(10.0));
    CHECK(y[2] == doctest::Approx(10.0));

}

TEST_CASE("sparse transpose") {
    // A = [1 0 2 ; 0 3 1]  (2x3), CSC:
    ippmm::SparseMatrix A(2, 3, {0, 1, 2, 4}, {0, 1, 0, 1}, {1.0, 3.0, 2.0, 1.0});
    ippmm::SparseMatrix At = A.transpose();

    // Aᵀ is [1 0 ; 0 3 ; 2 1]  (3x2)
    CHECK(At.rows() == 3);
    CHECK(At.cols() == 2);
    CHECK(At.nnz()  == 4);

    // Differential check: Aᵀx via the explicit transpose must equal
    // A.multiply_tr(x) (which computes Aᵀx without forming Aᵀ).
    for (const std::vector<double>& x : { std::vector<double>{5.0, 7.0},
                                          std::vector<double>{-1.0, 2.0} }) {
        const std::vector<double> viaT  = At.multiply(x);
        const std::vector<double> viaTr = A.multiply_tr(x);
        REQUIRE(viaT.size() == viaTr.size());
        for (std::size_t i = 0; i < viaT.size(); ++i)
            CHECK(viaT[i] == doctest::Approx(viaTr[i]));
    }
}