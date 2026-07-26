#pragma once

#include <vector>
#include "sparse_matrix.hpp"

namespace ippmm{

    // Dot product: x^T y
    Scalar dot(const std::vector<Scalar>& x, const std::vector<Scalar>& y);

    // AXPY, in place: y <- y + \alpha x
    void axpy(Scalar alpha, const std::vector<Scalar>& x, std::vector<Scalar>& y);

    // Vector norm (Default: ell-2 norm)
    enum class Norm{One, Two, Inf};
    Scalar norm(const std::vector<Scalar>& x, Norm type = Norm::Two);

    // Test a vector does not contain inf of NaN values
    bool all_finite(const std::vector<Scalar>& x);

}