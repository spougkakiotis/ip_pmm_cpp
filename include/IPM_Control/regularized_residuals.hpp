#pragma once

#include <vector>
#include "LinearAlgebra/sparse_matrix.hpp"  // Scalar, Int

namespace ippmm {

    // Apply IP-PMM's proximal regularization to the raw residuals:
    //   res_p = nr_res_p - delta * (y - lambda)
    //   res_d = nr_res_d + rho   * (x - zeta)
    // lambda (dual estimate) and zeta (primal estimate) are the PMM proximal centers.
    std::vector<Scalar> reg_primal_res(const std::vector<Scalar>& nr_res_p,
                                       Scalar delta,
                                       const std::vector<Scalar>& y,
                                       const std::vector<Scalar>& lambda);

    std::vector<Scalar> reg_dual_res(const std::vector<Scalar>& nr_res_d,
                                     Scalar rho,
                                     const std::vector<Scalar>& x,
                                     const std::vector<Scalar>& zeta);

}  