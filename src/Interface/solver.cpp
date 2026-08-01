#include "Interface/solver.hpp"

#include "IPM_Control/solver_state.hpp"      // SolverState, initialize_state
#include "IPM_Control/iteration.hpp"
#include "IPM_Control/stopping.hpp"
#include "IPM_Control/update_pmm.hpp"
#include "IPM_Control/regularized_residuals.hpp"
#include "IPM_Control/residuals.hpp"         // primal_residual, dual_residual
#include "KKT/kkt_system.hpp"
#include "KKT/kkt_solver.hpp"
#include "LinearAlgebra/vector_ops.hpp"

namespace ippmm {

    SolveResult solve(const QPProblem& qp, const SolveOptions& opts) {
        qp.validate();   // contract: presolved, consistent dimensions, no fixed vars


        // Warm start -> initial state (iterate, zeta/lambda, rho/delta, reg_limit, mu).
        SolverState s = initialize_state(qp, opts.tol);

        // Build the KKT machinery ONCE: fixed pattern + symbolic factorization.
        KKTSystem system(qp);
        KKTSolver solver(system.size(), system.col_ptr(), system.row_idx());

        SolveResult result;
        result.status = SolveStatus::MaxIterations;   // default if we exhaust maxit

        // "Old" non-regularized residuals for the current point.
        std::vector<Scalar> nr_res_p = primal_residual(qp, s.x);
        std::vector<Scalar> nr_res_d = dual_residual(qp, s.x, s.y, s.z_l, s.z_u);

        for (int iter = 0; iter < opts.maxit; ++iter) {
            // Stopping check on the current point.
            if (auto st = check_stopping(qp, nr_res_p, nr_res_d, s.mu, opts.tol)) {
                result.status = *st;
                result.iterations = iter;
                break;
            }

            // Regularized residuals for the direction (from the current point).
            const std::vector<Scalar> res_p = reg_primal_res(nr_res_p, s.delta, s.y, s.lambda);
            const std::vector<Scalar> res_d = reg_dual_res  (nr_res_d, s.rho,   s.x, s.zeta);

            // One IP-PMM step. Numerical failure -> terminate (retry logic deferred).
            const IterStatus ist = ip_pmm_iteration(s, qp, system, solver, res_p, res_d);
            if (ist == IterStatus::Instability) {
                result.status = SolveStatus::NumericalError;
                result.iterations = iter + 1;
                break;
            }

            // New residuals at the stepped point.
            std::vector<Scalar> new_nr_res_p = primal_residual(qp, s.x);
            std::vector<Scalar> new_nr_res_d = dual_residual(qp, s.x, s.y, s.z_l, s.z_u);

            // Accept/keep estimates and shrink regularization based on progress.
            update_pmm_parameters(s, nr_res_p, new_nr_res_p, nr_res_d, new_nr_res_d);
           
            // New becomes old for the next iteration's stopping check.
            nr_res_p = std::move(new_nr_res_p);
            nr_res_d = std::move(new_nr_res_d);
            result.iterations = iter + 1;
        }


        // Package the final iterate.
        result.x   = std::move(s.x);
        result.y   = std::move(s.y);
        result.z_l = std::move(s.z_l);
        result.z_u = std::move(s.z_u);
        return result;
    }

}  // namespace ippmm