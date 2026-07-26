#include "doctest.h"
#include "LinearAlgebra/vector_ops.hpp"
#include <limits>

using namespace ippmm;

TEST_CASE("dot product"){
    CHECK(dot({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}) == doctest::Approx(32.0));
}

TEST_CASE("axpy updates y in place") {
    std::vector<double> x{1.0, 2.0, 3.0};
    std::vector<double> y{10.0, 20.0, 30.0};
    axpy(2.0, x, y);                 // y <- 2*x + y
    REQUIRE(y.size() == 3);
    CHECK(y[0] == doctest::Approx(12.0));
    CHECK(y[1] == doctest::Approx(24.0));
    CHECK(y[2] == doctest::Approx(36.0));
}

TEST_CASE("vector norms") {
    const std::vector<double> x{-7.0, 3.0, -2.0};

    SUBCASE("2-norm is the default")  { CHECK(norm({3.0, 4.0}) == doctest::Approx(5.0)); }
    SUBCASE("explicit 2-norm")        { CHECK(norm({3.0, 4.0}, Norm::Two) == doctest::Approx(5.0)); }
    SUBCASE("1-norm")                 { CHECK(norm(x, Norm::One) == doctest::Approx(12.0)); } // 7+3+2
    SUBCASE("infinity norm")          { CHECK(norm(x, Norm::Inf) == doctest::Approx(7.0)); }
}

TEST_CASE("all_finite detects non-finite entries") {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    CHECK(all_finite({1.0, 2.0, 3.0}) == true);
    CHECK(all_finite({1.0, inf, 3.0}) == false);   // +inf caught
    CHECK(all_finite({1.0, -inf, 3.0}) == false);  // -inf caught
    CHECK(all_finite({1.0, nan, 3.0}) == false);   // NaN caught
}