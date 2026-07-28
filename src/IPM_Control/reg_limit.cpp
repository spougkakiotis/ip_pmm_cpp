#include "IPM_Control/reg_limit.hpp"

#include <algorithm>

namespace ippmm {
    Scalar reg_limit(const QPProblem& qp, Scalar tol, Scalar hard_lim) {
        const Scalar A_prod = qp.A.norm(SparseMatrix::MatNorm::One) * qp.A.norm(SparseMatrix::MatNorm::Inf);
        // Q is symmetric (upper-stored); its 1-norm and inf-norm are equal
        const Scalar qn = qp.Q.norm();  
        const Scalar Q_prod = qn*qn;

        const Scalar denom = std::max(Scalar(1.0), std::max(A_prod, Q_prod));
        Scalar base = 0.1 * std::min(tol, Scalar(1e-6)) / denom;
        base = std::max(base, hard_lim);

        const Scalar cap = (qp.Q.nnz() > 0) ? Scalar(1e-6) : Scalar(1e-8);
        return std::min(base, cap);
    }
}  