#include "residuals.hpp"

#include <cassert>
#include "vector_ops.hpp"

namespace ippmm{

    std::vector<Scalar> primal_residual(const QPProblem& qp,
                                        const std::vector<Scalar>& x){
        assert(static_cast<Int>(x.size()) == qp.num_vars());

        std::vector<Scalar> r = qp.A.multiply(x); 
        axpy(-1.0,qp.b,r);
        return r;
    }

    std::vector<Scalar> dual_residual(const QPProblem& qp,
                                      const std::vector<Scalar>& x,
                                      const std::vector<Scalar>& y,
                                      const std::vector<Scalar>& z){
        assert(static_cast<Int>(x.size()) == qp.num_vars());
        assert(static_cast<Int>(z.size()) == qp.num_vars());
        assert(static_cast<Int>(y.size()) == qp.num_constraints());

        std::vector<Scalar> r = qp.Q.multiply(x);
        axpy(1.0, qp.c, r);
        const std::vector<Scalar> Aty = qp.A.multiply_tr(y);
        axpy(-1.0, Aty, r);
        axpy(-1.0, z, r);
        return r;
    }

}