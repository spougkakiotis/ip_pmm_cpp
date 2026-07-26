#include <sparse_matrix.hpp>

#include <cassert>
#include <utility>

namespace ippmm{

    SparseMatrix::SparseMatrix(Int rows, Int cols,
                               std::vector<Int> col_ptr,
                               std::vector<Int> row_idx,
                               std::vector<Scalar> values) 
                : rows_(rows), cols_(cols), col_ptr_(std::move(col_ptr)),
                  row_idx_(std::move(row_idx)), values_(std::move(values)){

        // Test dimensions and well-definedness
        assert(rows_ >= 0 && cols_ >= 0);
        assert(static_cast<Int>(col_ptr_.size()) == cols_+1);
        assert(row_idx_.size() == values_.size());
        assert(col_ptr_.front() == 0);
        assert(col_ptr_.back() == static_cast<Int>(values_.size()));
    }

    std::vector<Scalar> SparseMatrix::multiply(const std::vector<Scalar>& x) const{
        assert(static_cast<Int>(x.size()) == cols_);

        std::vector<Scalar> y(rows_, 0.0);
        // This is a "scatter" routine (since we are in CSC format)
        for (Int j=0; j<cols_; ++j){
            const Scalar xj = x[j];
            for (Int k=col_ptr_[j]; k<col_ptr_[j+1]; ++k){
                y[row_idx_[k]] += values_[k]*xj;
            }
        }
        return y;
    }

    std::vector<Scalar> SparseMatrix::multiply_tr(const std::vector<Scalar>& x) const{
        assert(static_cast<Int>(x.size()) == rows_);

        std::vector<Scalar> y(cols_, 0.0);

        for (Int j=0; j<cols_; ++j){
            Scalar sum = 0.0;
            for (Int k=col_ptr_[j]; k<col_ptr_[j+1]; ++k){
                sum += values_[k]*x[row_idx_[k]];
            }
            y[j] = sum;
        }
        return y;
    }

    SparseMatrix SparseMatrix::transpose() const {
        // Aᵀ is (cols_ x rows_); its columns correspond to A's rows.
        std::vector<Int>    tp(rows_ + 1, 0);      // Aᵀ col_ptr, size rows_+1
        std::vector<Int>    ti(values_.size());    // Aᵀ row_idx
        std::vector<Scalar> tx(values_.size());    // Aᵀ values

        // Count nonzeros per row of A (= per column of Aᵀ).
        for (Int k = 0; k < static_cast<Int>(row_idx_.size()); ++k) {
            tp[row_idx_[k] + 1]++;
        }
        // Prefix-sum: turn counts into column start positions.
        for (Int r = 0; r < rows_; ++r) {
            tp[r + 1] += tp[r];
        }
        // Scatter each entry of A into its transposed slot.
        std::vector<Int> next(tp.begin(), tp.begin() + rows_);  // running cursor per Aᵀ column
        for (Int j = 0; j < cols_; ++j) {
            for (Int k = col_ptr_[j]; k < col_ptr_[j + 1]; ++k) {
                const Int r    = row_idx_[k];
                const Int dest = next[r]++;
                ti[dest] = j;             // A's column j becomes Aᵀ's row j
                tx[dest] = values_[k];
            }
        }
        return SparseMatrix(cols_, rows_, std::move(tp), std::move(ti), std::move(tx));
    }


}