#include "IPM_Control/stopping.hpp"

#include "LinearAlgebra/vector_ops.hpp"   // norm, Norm

namespace ippmm {

    std::optional<SolveStatus> check_stopping(const QPProblem& qp,
                                              const std::vector<Scalar>& nr_res_p,
                                              const std::vector<Scalar>& nr_res_d,
                                              Scalar mu,
                                              Scalar tol) {
        const Scalar p_inf = norm(nr_res_p, Norm::Inf);
        const Scalar d_inf = norm(nr_res_d, Norm::Inf);
        const Scalar b_inf = norm(qp.b,     Norm::Inf);
        const Scalar c_inf = norm(qp.c,     Norm::Inf);

        const bool primal_ok = p_inf / (1.0 + b_inf) < tol;
        const bool dual_ok   = d_inf / (1.0 + c_inf) < tol;
        const bool comp_ok   = mu < tol;

        if (primal_ok && dual_ok && comp_ok) {
            return SolveStatus::Optimal;
        }
        return std::nullopt;    // absence of a value
    }

}  // namespace ippmm