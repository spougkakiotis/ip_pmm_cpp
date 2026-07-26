#pragma once

#include <vector>
#include "qp_problem.hpp"
#include "sym_sparse_matrix.hpp"

namespace ippmm{

    class KKTSystem{
        public:
            explicit KKTSystem(const QPProblem& qp);

            Int size() const{return n_ + m_;}

            // Read-only access to the fixed sparsity pattern (KKTSolver symbolic step)
            const std::vector<Int>& col_ptr() const{return col_ptr_;}
            const std::vector<Int>& row_idx() const{return row_idx_;}

            //To do assemble()

        private:
            Int n_ = 0; 
            Int m_ = 0;

            // Fixed upper-triangular pattern of K
            std::vector<Int> col_ptr_;  // size (n+m)+1 
            std::vector<Int> row_idx_;  // size nnz(upper K)

            // Per-slot provenance so assemble() knows what each value slot holds

            //Filled alongside the sparsity pattern
            SparseMatrix At_;
            std::vector<Int> x_diag_slot_;  
    };

}