#include "IPM_Control/warm_start.hpp"
#include <cassert>
#include <cmath>
#include <algorithm>
#include "LinearAlgebra/sym_sparse_matrix.hpp"
#include "LinearAlgebra/vector_ops.hpp"
#include "KKT/kkt_solver.hpp"

namespace ippmm {
    namespace {

        // Assemble the UPPER TRIANGLE of  M = [ I     Aᵀ  ]   (n+m square, quasidefinite)
        //                                     [ A   -δ₀ I ]
        // as a SymSparseMatrix. Columns 0..n-1 hold the identity diagonal; columns
        // n..n+m-1 hold A's row k (= Aᵀ column k) above a -δ₀ diagonal.
        SymSparseMatrix assemble_M(const QPProblem& qp, Scalar delta0) {
            const Int n = qp.num_vars();
            const Int m = qp.num_constraints();
            const Int total = n + m;

            const SparseMatrix At = qp.A.transpose();   // columns of At are rows of A
            const auto& Atp = At.col_ptr();
            const auto& Ati = At.row_idx();
            const auto& Atx = At.values();

            std::vector<Int> col_ptr(total + 1, 0);

            // Count: x-columns have 1 (identity diagonal); y-columns have nnz(A row k) + 1.
            for (Int j = 0; j < n; ++j) col_ptr[j + 1] = 1;
            for (Int k = 0; k < m; ++k) col_ptr[n + k + 1] = (Atp[k + 1] - Atp[k]) + 1;
            for (Int c = 0; c < total; ++c) col_ptr[c + 1] += col_ptr[c];

            std::vector<Int>    row_idx(col_ptr[total]);
            std::vector<Scalar> values(col_ptr[total]);
            std::vector<Int>    next(col_ptr.begin(), col_ptr.begin() + total);

            // x-columns: identity diagonal (1) at (j, j).
            for (Int j = 0; j < n; ++j) {
                row_idx[next[j]] = j;
                values[next[j]]  = 1.0;
                ++next[j];
            }
            // y-columns: Aᵀ column k (rows < n), then -δ₀ on the diagonal (n+k, n+k).
            for (Int k = 0; k < m; ++k) {
                const Int col = n + k;
                for (Int t = Atp[k]; t < Atp[k + 1]; ++t) {
                    row_idx[next[col]] = Ati[t];
                    values[next[col]]  = Atx[t];
                    ++next[col];
                }
                row_idx[next[col]] = col;
                values[next[col]]  = -delta0;
                ++next[col];
            }

            return SymSparseMatrix(total, std::move(col_ptr), std::move(row_idx), std::move(values));
        }

    }  // namespace

    WarmStartPoint warm_start_point(const QPProblem& qp, Scalar delta0) {
        const Int n = qp.num_vars();
        const Int m = qp.num_constraints();

        // Factor M once.
        const SymSparseMatrix M = assemble_M(qp, delta0);
        KKTSolver solver(M.size(), M.col_ptr(), M.row_idx());
        const bool ok = solver.factorize(M.values());
        assert(ok && "warm-start M factorization failed");
        (void)ok;   // avoid compiler warnings

        // Solve 1:  M [x; w] = [0; b]   ->  x is the top block.
        std::vector<Scalar> rhs1(n + m, 0.0);
        for (Int k = 0; k < m; ++k) rhs1[n + k] = qp.b[k];
        const std::vector<Scalar> sol1 = solver.solve(rhs1);
        std::vector<Scalar> x(sol1.begin(), sol1.begin() + n);

        // Solve 2:  M [·; y] = [0; A(c + Qx)]   ->  y is the bottom block.
        std::vector<Scalar> cqx = qp.Q.multiply(x);   // Q x
        axpy(1.0, qp.c, cqx);                          // c + Q x
        const std::vector<Scalar> Acqx = qp.A.multiply(cqx);  // A(c + Qx)
        std::vector<Scalar> rhs2(n + m, 0.0);
        for (Int k = 0; k < m; ++k) rhs2[n + k] = Acqx[k];
        const std::vector<Scalar> sol2 = solver.solve(rhs2);
        std::vector<Scalar> y(sol2.begin() + n, sol2.end());

        // z = c + Q x - Aᵀ y ; split into z_l / z_u per bound side.
        std::vector<Scalar> z = qp.Q.multiply(x);
        axpy(1.0, qp.c, z);
        const std::vector<Scalar> Aty = qp.A.multiply_tr(y);
        axpy(-1.0, Aty, z);

        WarmStartPoint p;
        p.x = std::move(x);
        p.y = std::move(y);
        p.z_l.assign(n, 0.0);
        p.z_u.assign(n, 0.0);
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) p.z_l[i] = std::max( z[i], Scalar(0.0));  // positive part
            if (qp.has_upper(i)) p.z_u[i] = std::max(-z[i], Scalar(0.0));  // negative part
        }
        return p;
    }


    void finalize_start(const QPProblem& qp, WarmStartPoint& p, Scalar kappa) {
        const Int n = qp.num_vars();
        auto& x = p.x; auto& z_l = p.z_l; auto& z_u = p.z_u;

        // ---- Push near-boundary slacks / multipliers off the wall ----
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i)) {
                if (x[i] - qp.lb[i] <= 1e-4) x[i]   = qp.lb[i] + 0.1;
                if (z_l[i]          <= 1e-4) z_l[i] = 0.1;
            }
            if (qp.has_upper(i)) {
                if (qp.ub[i] - x[i] <= 1e-4) x[i]   = qp.ub[i] - 0.1;
                if (z_u[i]          <= 1e-4) z_u[i] = 0.1;
            }
        }

        // ---- Boxed primal x -> kappa-clamp into the interior band ----
        for (Int i = 0; i < n; ++i) {
            if (qp.has_lower(i) && qp.has_upper(i)) {
                const Scalar margin = kappa * (qp.ub[i] - qp.lb[i]);
                x[i] = std::min(std::max(x[i], qp.lb[i] + margin), qp.ub[i] - margin);
            }
        }

        // ---- Mehrotra shift over one group.
        //      Returns {delta_x_bar, delta_z_bar} for the given group/side. ----
        auto mehrotra = [&](auto active, bool lower) -> std::pair<Scalar, Scalar> {
            Int num = 0;
            Scalar min_g = std::numeric_limits<Scalar>::infinity();
            Scalar min_z = std::numeric_limits<Scalar>::infinity();
            for (Int i = 0; i < n; ++i) if (active(i)) {
                const Scalar g = lower ? (x[i] - qp.lb[i]) : (qp.ub[i] - x[i]);
                const Scalar z = lower ? z_l[i] : z_u[i];
                min_g = std::min(min_g, g); min_z = std::min(min_z, z); ++num;
            }
            if (num == 0) return {0.0, 0.0};
            const Scalar dx = std::max(-1.5 * min_g, Scalar(0.0));
            const Scalar dz = std::max(-1.5 * min_z, Scalar(0.0));
            Scalar tp = 0.0, sum_z = 0.0, sum_g = 0.0;
            for (Int i = 0; i < n; ++i) if (active(i)) {
                const Scalar g = lower ? (x[i] - qp.lb[i]) : (qp.ub[i] - x[i]);
                const Scalar z = lower ? z_l[i] : z_u[i];
                tp += (g + dx) * (z + dz);  sum_z += z;  sum_g += g;
            }
            const Scalar denom_x = sum_z + num * dz;      // cross-coupling: primal uses dz
            const Scalar denom_z = sum_g + num * dx;      //                 dual uses dx
            const Scalar dx_bar = dx + (denom_x > 0.0 ? 0.5 * tp / denom_x : 0.0);
            const Scalar dz_bar = dz + (denom_z > 0.0 ? 0.5 * tp / denom_z : 0.0);
            return {dx_bar, dz_bar};
        };

        auto one_sided_lower = [&](Int i){ return qp.has_lower(i) && !qp.has_upper(i); };
        auto one_sided_upper = [&](Int i){ return qp.has_upper(i) && !qp.has_lower(i); };
        auto lower_active    = [&](Int i){ return qp.has_lower(i); };
        auto upper_active    = [&](Int i){ return qp.has_upper(i); };

        // ---- Compute ALL four magnitudes on the current point ----
        const Scalar dxl_bar = mehrotra(one_sided_lower, true ).first;   // primal, one-sided
        const Scalar dxu_bar = mehrotra(one_sided_upper, false).first;
        const Scalar dzl_bar = mehrotra(lower_active,    true ).second;  // dual, all-active
        const Scalar dzu_bar = mehrotra(upper_active,    false).second;

        // ---- Apply. Boxed x is NOT primal-shifted (already clamped);
        //      boxed z IS dual-shifted (it's in lower_active and upper_active). ----
        for (Int i = 0; i < n; ++i) {
            if (one_sided_lower(i)) x[i]   += dxl_bar;
            if (one_sided_upper(i)) x[i]   -= dxu_bar;
            if (qp.has_lower(i))    z_l[i] += dzl_bar;
            if (qp.has_upper(i))    z_u[i] += dzu_bar;
        }
    }
}  // namespace ippmm