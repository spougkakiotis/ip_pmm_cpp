#include "doctest.h"
#include "Interface/solver_status.hpp"
#include <string>

using namespace ippmm;

TEST_CASE("SolveResult defaults and status strings") {
    SolveResult r;
    CHECK(r.status == SolveStatus::Undefined);   // safe default
    CHECK(r.iterations == 0);
    CHECK(r.x.empty());

    CHECK(std::string(to_string(SolveStatus::Optimal)) == "optimal");
    CHECK(std::string(to_string(SolveStatus::DualInfeasible)) == "dual infeasible");
}