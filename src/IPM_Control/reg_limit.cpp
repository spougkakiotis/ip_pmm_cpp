#include "IPM_Control/reg_limit.hpp"

#include <algorithm>

namespace ippmm {
    Scalar reg_limit(const QPProblem& qp, Scalar tol) {
        const Scalar A_prod = qp.A.norm(SparseMatrix::MatNorm::One)
                            * qp.A.norm(SparseMatrix::MatNorm::Inf);
        const Scalar qn     = qp.Q.norm();
        const Scalar Q_prod = qn * qn;

        const bool   is_qp    = (qp.Q.nnz() > 0);
        const Scalar hard_lim = is_qp ? Scalar(5e-8) : Scalar(5e-10);
        const Scalar cap      = is_qp ? Scalar(1e-6) : Scalar(1e-8);

        const Scalar denom = std::max(Scalar(1.0), std::max(A_prod, Q_prod));
        Scalar base = 0.1 * std::min(tol, Scalar(1e-6)) / denom;
        base = std::max(base, hard_lim);

        return std::min(base, cap);
    }
}  