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

    


}