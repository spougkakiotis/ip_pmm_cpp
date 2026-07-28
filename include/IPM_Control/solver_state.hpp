#pragma once

#include <vector>
#include "LinearAlgebra/sparse_matrix.hpp"

namespace ippmm{

    struct SolverState{
        // Current iterate
        std::vector<Scalar> x, y, z_l, z_u;

        //PMM proximal centers
        std::vector<Scalar> zeta;
        std::vector<Scalar> lambda;

        // Regularization parameters
        Scalar rho = 0.0;   // primal
        Scalar delta = 0.0; // dual
        Scalar reg_limit = 0.0;

        // Barrier/centrality bookkeeping
        Scalar mu = 0.0;        // duality measure
        Scalar mu_rate = 0.0;   // drives rho/delta decrease rate

        // Non-regularized residual norms (latest iteration)
        Scalar nr_res_p_norm = 0.0;
        Scalar nr_res_d_norm = 0.0;

        // Counters 
        int no_primal_update = 0;
        int no_dual_update   = 0;
    };

}