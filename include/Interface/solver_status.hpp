#pragma once

#include <vector>
#include "LinearAlgebra/sparse_matrix.hpp"  // ippmm::Scalar, Int

namespace ippmm {

    // Termination status cases:
    enum class SolveStatus {
        Optimal,
        MaxIterations,
        PrimalInfeasible,
        DualInfeasible,
        NumericalError,
        InsufficientAccuracy,
        Undefined
    };

    // The solver's return bundle: primal x, multipliers y, and the two bound-dual
    // vectors z_l (lower) and z_u (upper), plus status and iteration count.
    struct SolveResult {
        std::vector<Scalar> x;      // primal solution
        std::vector<Scalar> y;      // Lagrange multipliers (equality constraints)
        std::vector<Scalar> z_l;    // lower-bound dual multipliers (>= 0)
        std::vector<Scalar> z_u;    // upper-bound dual multipliers (>= 0)
        SolveStatus status = SolveStatus::Undefined;
        int iterations = 0;
    };

    // Human-readable status, for logging/printing.
    const char* to_string(SolveStatus s);

}  