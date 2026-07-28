#include "doctest.h"
#include "IPM_Control/solver_state.hpp"

using namespace ippmm;

TEST_CASE("SolverState default-constructs to a clean zero state") {
    SolverState s;
    CHECK(s.rho == 0.0);
    CHECK(s.delta == 0.0);
    CHECK(s.mu == 0.0);
    CHECK(s.no_primal_update == 0);
    CHECK(s.x.empty());
    CHECK(s.zeta.empty());
}