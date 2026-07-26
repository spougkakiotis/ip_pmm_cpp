#include "kkt_system.hpp"

#include <cassert>

namespace ippmm{

    KKTSystem::KKTSystem(const QPProblem& qp)
             : n_(qp.num_vars()),
               m_(qp.num_constraints()),
               At_(qp.A.transpose()),
               x_diag_slot_(qp.num_vars(),-1){

        const Int total = n_ + m_;
        col_ptr_.assign(total+1,0);

        const auto& Qp = qp.Q.col_ptr();
        const auto& Qi = qp.Q.row_idx();
        const auto& Atp = At_.col_ptr();
        const auto& Ati = At_.row_idx();

        // Count nonzeros per column of the upper triangle of K. 
        // x-columns (0..n-1): upper part of Q's column j, plus a guaranteed diagonal.
        for (Int j = 0; j < n_; ++j) {
            Int cnt = 0;
            bool has_diag = false;
            for (Int k = Qp[j]; k < Qp[j + 1]; ++k) {
                const Int r = Qi[k];       // Q is upper-stored, so r <= j
                cnt++;
                if (r == j) has_diag = true;
            }
            if (!has_diag) cnt++;          // reserve the (j,j) slot (merge/insert)
            col_ptr_[j + 1] = cnt;
        }

        // y-columns (n..n+m-1): column k of Aᵀ (rows < n), plus a δ diagonal.
        for (Int k = 0; k < m_; ++k) {
            const Int col = n_ + k;
            const Int a_nnz = Atp[k + 1] - Atp[k];
            col_ptr_[col + 1] = a_nnz + 1;   // + the diagonal δ slot
        }

        // Prefix-sum counts into start offsets.
        for (Int c = 0; c < total; ++c) col_ptr_[c + 1] += col_ptr_[c];

        // Fill row indices in the fixed layout. ---
        row_idx_.assign(col_ptr_[total], 0);
        std::vector<Int> next(col_ptr_.begin(), col_ptr_.begin() + total);

        // x-columns: merge Q's upper entries with the guaranteed diagonal, in row order.
        for (Int j = 0; j < n_; ++j) {
            bool diag_written = false;
            for (Int k = Qp[j]; k < Qp[j + 1]; ++k) {
                const Int r = Qi[k];
                if (r == j) {                       // Q already has the diagonal
                    x_diag_slot_[j] = next[j];
                    diag_written = true;
                }
                row_idx_[next[j]++] = r;
            }
            if (!diag_written) {                    // insert the diagonal (Q lacked it)
                x_diag_slot_[j] = next[j];
                row_idx_[next[j]++] = j;
            }
        }
        // y-columns: Aᵀ's column k (row indices < n), then the diagonal (n+k).
        for (Int k = 0; k < m_; ++k) {
            const Int col = n_ + k;
            for (Int t = Atp[k]; t < Atp[k + 1]; ++t) {
                row_idx_[next[col]++] = Ati[t];     // these are < n (A's column indices)
            }
            row_idx_[next[col]++] = col;            // δ diagonal at (n+k, n+k)
        }
        // Sanity: every slot was filled exactly once (each cursor reached its column end).
        for (Int c = 0; c < total; ++c) assert(next[c] == col_ptr_[c + 1]);
    }
            

}