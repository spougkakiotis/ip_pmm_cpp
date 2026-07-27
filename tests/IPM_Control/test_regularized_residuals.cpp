#include "doctest.h"
#include "IPM_Control/regularized_residuals.hpp"

using namespace ippmm;

TEST_CASE("regularize_primal: res_p = nr_res_p - delta*(y - lambda)") {
    // nr_res_p=[1,2], delta=0.5, y=[4,6], lambda=[2,2]
    // y-lambda=[2,4]; delta*(...)=[1,2]; res_p=[1-1, 2-2]=[0,0]
    const std::vector<double> res =
        reg_primal_res({1.0, 2.0}, 0.5, {4.0, 6.0}, {2.0, 2.0});
    REQUIRE(res.size() == 2);
    CHECK(res[0] == doctest::Approx(0.0));
    CHECK(res[1] == doctest::Approx(0.0));
}

TEST_CASE("regularize_dual: res_d = nr_res_d + rho*(x - zeta)") {
    // nr_res_d=[3,1,0], rho=2, x=[1,1,1], zeta=[0,2,1]
    // x-zeta=[1,-1,0]; rho*(...)=[2,-2,0]; res_d=[3+2, 1-2, 0+0]=[5,-1,0]
    const std::vector<double> res =
        reg_dual_res({3.0, 1.0, 0.0}, 2.0, {1.0, 1.0, 1.0}, {0.0, 2.0, 1.0});
    REQUIRE(res.size() == 3);
    CHECK(res[0] == doctest::Approx(5.0));
    CHECK(res[1] == doctest::Approx(-1.0));
    CHECK(res[2] == doctest::Approx(0.0));
}

TEST_CASE("zero regularization is identity") {
    // rho=delta=0 -> residuals unchanged
    const std::vector<double> rp = reg_primal_res({7.0, -3.0}, 0.0, {1.0, 1.0}, {9.0, 9.0});
    CHECK(rp[0] == doctest::Approx(7.0));
    CHECK(rp[1] == doctest::Approx(-3.0));
}