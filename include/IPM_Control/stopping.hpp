#pragma once

#include <optional>
#include <vector>
#include "qp_problem.hpp"
#include "Interface/solver_status.hpp"

namespace ippmm {

    // Optimality test (inf-norm, relative):
    //   ||nr_res_p||_inf / (1 + ||b||_inf) < tol
    //   && ||nr_res_d||_inf / (1 + ||c||_inf) < tol
    //   && mu < tol
    // Returns SolveStatus::Optimal if all hold, else std::nullopt (keep iterating).
    // Infeasibility branches are intentionally omitted for now.
    std::optional<SolveStatus> check_stopping(const QPProblem& qp,
                                              const std::vector<Scalar>& nr_res_p,
                                              const std::vector<Scalar>& nr_res_d,
                                              Scalar mu,
                                              Scalar tol);

}  