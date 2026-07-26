#include "LinearAlgebra/vector_ops.hpp"
#include <cassert>
#include <cmath>

namespace ippmm{
    Scalar dot(const std::vector<Scalar>& x, const std::vector<Scalar>& y){
        assert(x.size() == y.size());
        Scalar sum = 0.0;
        for (std::size_t i=0; i<x.size(); ++i){
            sum += x[i]*y[i];
        }
        return sum;
    }

    void axpy(Scalar alpha, const std::vector<Scalar>& x, std::vector<Scalar>& y){
        assert(x.size() == y.size());
        for (std::size_t i=0; i<x.size(); ++i){
            y[i] += alpha*x[i];
        }
    }

    Scalar norm(const std::vector<Scalar>& x, Norm type){
        Scalar result = 0.0;
        switch(type){
            case Norm::One:
                for (Scalar xi : x) result += std::abs(xi);
                break;
            case Norm::Two:
                result = std::sqrt(dot(x,x));
                assert(std::isfinite(result) && "norm overflow/NaN — inputs out of range");
                break;
            case Norm::Inf:
                for (Scalar xi : x) result = std::max(result, std::abs(xi));
                break;
        }
        return result;
    }

    bool all_finite(const std::vector<Scalar>& x){
        for (Scalar xi : x) if (!std::isfinite(xi)) return false;
        return true;
    }

}