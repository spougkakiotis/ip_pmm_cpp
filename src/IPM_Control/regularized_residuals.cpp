#include "IPM_Control/regularized_residuals.hpp"

#include <cassert>
#include "LinearAlgebra/vector_ops.hpp"   // axpy

namespace ippmm {

    std::vector<Scalar> reg_primal_res(const std::vector<Scalar>& nr_res_p,
                                       Scalar delta,
                                       const std::vector<Scalar>& y,
                                       const std::vector<Scalar>& lambda) {
        assert(nr_res_p.size() == y.size() && y.size() == lambda.size());

        // res_p = nr_res_p - delta*(y - lambda)
        std::vector<Scalar> res = nr_res_p;
        axpy(-delta,  y,      res);   // res -= delta*y
        axpy( delta,  lambda, res);   // res += delta*lambda
        return res;
    }

    std::vector<Scalar> reg_dual_res(const std::vector<Scalar>& nr_res_d,
                                     Scalar rho,
                                     const std::vector<Scalar>& x,
                                     const std::vector<Scalar>& zeta) {
        assert(nr_res_d.size() == x.size() && x.size() == zeta.size());

        // res_d = nr_res_d + rho*(x - zeta)
        std::vector<Scalar> res = nr_res_d;
        axpy( rho, x,    res);   // res += rho*x
        axpy(-rho, zeta, res);   // res -= rho*zeta
        return res;
    }

}  