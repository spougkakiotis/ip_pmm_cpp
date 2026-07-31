#include "IPM_Control/residuals.hpp"
#include <cassert>
#include "LinearAlgebra/vector_ops.hpp"

namespace ippmm{

    std::vector<Scalar> primal_residual(const QPProblem& qp,
                                        const std::vector<Scalar>& x){
        assert(static_cast<Int>(x.size()) == qp.num_vars());

        std::vector<Scalar> r = qp.A.multiply(x); 
        for (std::size_t i = 0; i < r.size(); ++i) r[i] = qp.b[i] - r[i];
        return r;
    }

    std::vector<Scalar> dual_residual(const QPProblem& qp,
                                    const std::vector<Scalar>& x,
                                    const std::vector<Scalar>& y,
                                    const std::vector<Scalar>& z_l,
                                    const std::vector<Scalar>& z_u) {
        assert(static_cast<Int>(x.size())   == qp.num_vars());
        assert(static_cast<Int>(y.size())   == qp.num_constraints());
        assert(static_cast<Int>(z_l.size()) == qp.num_vars());
        assert(static_cast<Int>(z_u.size()) == qp.num_vars());

        std::vector<Scalar> r = qp.Q.multiply(x);       // r = Q x
        axpy( 1.0, qp.c, r);                             // r += c
        const std::vector<Scalar> Aty = qp.A.multiply_tr(y);
        axpy(-1.0, Aty, r);                             // r -= Aᵀy
        axpy(-1.0, z_l, r);                             // r -= z_l
        axpy( 1.0, z_u, r);                             // r += z_u
        return r;
    }

}