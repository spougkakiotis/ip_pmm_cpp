#pragma once

#include <cstddef>
#include <vector>

namespace ippmm{

    using Int = int;    // index type for sparse structure
    using Scalar = double;  // numeric type

    class SparseMatrix{
        public:
            SparseMatrix(Int rows, Int cols,
                         std::vector<Int> col_ptr,
                         std::vector<Int> row_idx,
                         std::vector<Scalar> values);
            
            Int rows() const {return rows_;}
            Int cols() const {return cols_;}
            Int nnz()  const {return static_cast<Int>(values_.size());}

            // Returns y = A*x
            std::vector<Scalar> multiply(const std::vector<Scalar>& x) const;

            // Read-only access to the raw CSC arrays 
            const std::vector<Int>&    col_ptr() const {return col_ptr_;}
            const std::vector<Int>&    row_idx() const {return row_idx_;}
            const std::vector<Scalar>& values() const{return values_;}

        private:
                Int rows_ = 0;
                Int cols_ = 0;
                std::vector<Int>    col_ptr_; // size cols_ + 1
                std::vector<Int>    row_idx_; // size nnz
                std::vector<Scalar> values_;  // size nnz
    };



}   // namespace ippmm