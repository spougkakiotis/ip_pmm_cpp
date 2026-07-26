#include "kkt_solver.hpp"

#include <cassert>
#include <stdexcept>
#include <utility>

// QDLDL is a C library: give its declarations C linkage so the C++ linker
// resolves them against the C-compiled qdldl.o.
extern "C" {
#include "qdldl.h"
}

namespace ippmm {

    KKTSolver::KKTSolver(Int n,
                        std::vector<Int> col_ptr,
                        std::vector<Int> row_idx)
            : n_(n),
              Ap_(std::move(col_ptr)),
              Ai_(std::move(row_idx)) {
        assert(n_ > 0);
        assert(static_cast<Int>(Ap_.size()) == n_ + 1);
        assert(Ap_.front() == 0);
        assert(Ap_.back() == static_cast<Int>(Ai_.size()));

        // --- Symbolic tier: elimination tree, computed once. ---
        etree_.resize(n_);
        Lnz_.resize(n_);
        std::vector<Int> work(n_);  // scratch, needed only here

        sum_Lnz_ = QDLDL_etree(n_, Ap_.data(), Ai_.data(),
                               work.data(), Lnz_.data(), etree_.data());

        // QDLDL_etree returns -1 if the input is not upper-triangular / has a
        // structurally invalid pattern for LDL. That's a caller error, so throw.
        if (sum_Lnz_ < 0) {
            throw std::invalid_argument(
                "KKTSolver: invalid sparsity pattern (etree failed; "
                "is the matrix upper-triangular with no duplicate indices?)");
        }

        // Size the numeric + scratch buffers now that we know sum_Lnz_.
        Lp_.resize(n_ + 1);
        Li_.resize(sum_Lnz_);
        Lx_.resize(sum_Lnz_);
        D_.resize(n_);
        Dinv_.resize(n_);

        bwork_.resize(n_);
        iwork_.resize(3 * n_);
        fwork_.resize(n_);
    }

    bool KKTSolver::factorize(const std::vector<Scalar>& values) {
        assert(static_cast<Int>(values.size()) == static_cast<Int>(Ai_.size()));

        const Int pos = QDLDL_factor(
            n_, Ap_.data(), Ai_.data(), values.data(),
            Lp_.data(), Li_.data(), Lx_.data(), D_.data(), Dinv_.data(),
            Lnz_.data(), etree_.data(),
            bwork_.data(), iwork_.data(), fwork_.data());

        factorized_ = (pos >= 0);   // -1 => zero pivot => not factorizable
        return factorized_;
    }

    std::vector<Scalar> KKTSolver::solve(const std::vector<Scalar>& rhs) const {
        assert(factorized_ && "solve() called before a successful factorize()");
        assert(static_cast<Int>(rhs.size()) == n_);

        // QDLDL_solve works in place: x starts as b, ends as the solution.
        std::vector<Scalar> x = rhs;  // copy in
        QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), x.data());
        return x;
    }

}  // namespace ippmm