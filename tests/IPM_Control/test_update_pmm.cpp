#include "doctest.h"
#include "IPM_Control/update_pmm.hpp"

using namespace ippmm;

TEST_CASE("update_pmm: progress -> accept estimates, aggressive shrink") {
    SolverState s;
    s.x = {1.0, 2.0}; s.y = {3.0};
    s.zeta = {0.0, 0.0}; s.lambda = {0.0};
    s.rho = 8.0; s.delta = 8.0; s.reg_limit = 1e-8; s.mu_rate = 0.5;

    // Residuals dropped a lot (new << 0.95*old) -> cond true.
    update_pmm_parameters(s, /*nr_p=*/{10.0}, /*new_p=*/{1.0},
                             /*nr_d=*/{10.0}, /*new_d=*/{1.0});

    CHECK(s.zeta == s.x);          // accepted primal estimate
    CHECK(s.lambda == s.y);        // accepted dual estimate
    CHECK(s.rho   == doctest::Approx(8.0 * (1.0 - 0.5)));   // 4.0
    CHECK(s.delta == doctest::Approx(8.0 * (1.0 - 0.5)));   // 4.0
    CHECK(s.no_primal_update == 0);
    CHECK(s.no_dual_update   == 0);
}

TEST_CASE("update_pmm: no progress -> keep estimates, slow shrink, bump counters") {
    SolverState s;
    s.x = {1.0}; s.y = {2.0};
    s.zeta = {9.0}; s.lambda = {9.0};        // stale estimates, must stay
    s.rho = 8.0; s.delta = 8.0; s.reg_limit = 1e-8; s.mu_rate = 0.5;

    // Residuals barely moved (new > 0.95*old) -> cond false.
    update_pmm_parameters(s, {10.0}, {9.9}, {10.0}, {9.9});

    CHECK(s.zeta[0]   == 9.0);      // estimate unchanged
    CHECK(s.lambda[0] == 9.0);
    CHECK(s.rho   == doctest::Approx(8.0 * (1.0 - 0.666 * 0.5)));   // slow
    CHECK(s.delta == doctest::Approx(8.0 * (1.0 - 0.666 * 0.5)));
    CHECK(s.no_primal_update == 1);
    CHECK(s.no_dual_update   == 1);
}

TEST_CASE("update_pmm: shrink is floored at reg_limit") {
    SolverState s;
    s.x = {1.0}; s.y = {1.0}; s.zeta = {0.0}; s.lambda = {0.0};
    s.rho = 1e-8; s.delta = 1e-8; s.reg_limit = 1e-8; s.mu_rate = 0.9;
    // cond true, but rho*full would go below reg_limit -> clamped.
    update_pmm_parameters(s, {10.0}, {0.0}, {10.0}, {0.0});
    CHECK(s.rho   == doctest::Approx(1e-8));   // floored
    CHECK(s.delta == doctest::Approx(1e-8));
}