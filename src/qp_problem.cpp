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
        if (static_cast<Int>(is_free.size()) != n)
            throw std::invalid_argument("QPProblem: is_free length != n");
        if (Q.rows() != n || Q.cols() != n)
            throw std::invalid_argument("QPProblem: Q must be n x n");
    }
}