#include "qp_problem.hpp"

#include <stdexcept>

namespace ippmm{

    void QPProblem::validate() const{
        const Int n = num_vars();
        const Int m = num_constraints();

        if (static_cast<Int>(c.size()) != n)
            throw std::invalid_argument("QPProblem: c length != n");
        if (static_cast<Int>(b.size()) != m)
            throw std::invalid_argument("QPProblem: b length != m");
        if (static_cast<Int>(lb.size()) != n)
            throw std::invalid_argument("QPProblem: lb length != n");
        if (static_cast<Int>(ub.size()) != n)
            throw std::invalid_argument("QPProblem: ub length != n");
        for (Int i = 0; i < n; ++i) {
            if (lb[i] > ub[i])
                throw std::invalid_argument("QPProblem: crossed bounds (lb > ub)");
        }
        if (Q.size() != n)
            throw std::invalid_argument("QPProblem: Q must be n x n");
    }
}