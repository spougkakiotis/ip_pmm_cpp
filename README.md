# IP-PMM (C++)

A C++ implementation of an **Interior Point–Proximal Method of Multipliers (IP-PMM)**
for convex quadratic programming with general two-sided bounds. The solver is built around
sparse linear algebra and a permissively licensed sparse factorization.

The solver targets problems of the form

```
minimize     cᵀx + ½ xᵀ Q x
subject to   A x = b
             lb ≤ x ≤ ub
```

where `A` is `m × n`, `Q` is symmetric positive semidefinite (stored upper-triangular),
and each bound may be finite or infinite. `lb_i = -∞` and `ub_i = +∞` makes variable `i`
free; a one-sided bound uses an infinite value on the unconstrained side. This
formulation subsumes the classic `x ≥ 0` standard form as a special case.

> **Status: a working solver.** The core algorithm solves convex QPs end to end and
> is verified against known solutions. Robustness hardening (retry on ill-conditioning,
> infeasibility detection) and real-problem input/validation are in progress — see the
> [Roadmap](#roadmap).

---

## Design goals

- **Fully permissive licensing.** MIT-licensed, depending only on permissively
  licensed components (see [Licensing](#licensing)); no GPL/LGPL dependencies.
- **Sparse throughout.** Problem data and the Newton systems are stored and factored
  in compressed sparse column (CSC) form.
- **Quasidefinite by construction.** IP-PMM's primal–dual proximal regularization
  makes each Newton (KKT) system quasidefinite, admitting a pivot-free LDLᵀ
  factorization — matched exactly by the QDLDL backend.
- **Built bottom-up and tested.** Every kernel has unit tests with hand-computed
  oracles; the assembled solver is checked end to end against known QP solutions.

---

## Requirements

- A C++20 compiler (developed with `g++` 13+)
- GNU `make`
- `git` and `curl` (used once by `setup.sh` to fetch dependencies)

Developed on Ubuntu (native or via WSL2 on Windows). Any Unix-like toolchain works.

---

## Getting started

```bash
git clone git@github.com:spougkakiotis/ip_pmm_cpp.git
cd ip_pmm_cpp

./setup.sh        # fetch third-party deps into third_party/ (gitignored)
make              # build the solver binary   -> build/ip_pmm
make test         # build and run the test suite
```

`setup.sh` fetches the header-only test framework (doctest) and the QDLDL
factorization library, and writes the two small headers QDLDL normally generates
via CMake. Everything it fetches lives under `third_party/` (gitignored), so the
dependencies are reconstructed from `setup.sh` rather than committed.

### Make targets

| Target       | Effect                                              |
| ------------ | --------------------------------------------------- |
| `make`       | Build the solver executable (`build/ip_pmm`)        |
| `make run`   | Build, then run the solver                          |
| `make test`  | Build and run the full unit-test suite              |
| `make clean` | Remove all build artifacts (`build/`)               |

Builds use `-std=c++20 -Wall -Wextra -Wpedantic` at `-O0 -g` for debuggable
development. Header dependencies are tracked automatically (`-MMD -MP`). Vendored
QDLDL (C) is compiled with `gcc`; everything else is C++.

---

## Using the solver

```cpp
#include "Interface/solver.hpp"
using namespace ippmm;

// Build A (CSC), Q (upper-triangular symmetric), c, b, lb, ub ...
QPProblem qp{c, A, Q, b, lb, ub};

SolveOptions opts;          // opts.tol = 1e-4, opts.maxit = 100 by default
SolveResult r = solve(qp, opts);

if (r.status == SolveStatus::Optimal) {
    // r.x, r.y, r.z_l, r.z_u hold the primal solution and multipliers;
    // r.iterations is the iteration count.
}
```

`solve` returns a `SolveResult` bundling the primal solution `x`, equality
multipliers `y`, the lower/upper bound multipliers `z_l`/`z_u`, a `SolveStatus`,
and the iteration count. The problem must be presolved (no fixed variables, i.e.
`lb_i == ub_i`); `qp.validate()` enforces this contract.

---

## Project layout

Sources are grouped by module, mirrored across `include/`, `src/`, and `tests/`.

```
ip_pmm_cpp/
├── include/
│   ├── LinearAlgebra/     sparse_matrix, sym_sparse_matrix, vector_ops
│   ├── KKT/               kkt_system, kkt_solver, newton_backsolve
│   ├── IPM_Control/       residuals, complementarity, ratio_test,
│   │                      regularized_residuals, predictor_corrector,
│   │                      boundary_control, reg_limit, warm_start,
│   │                      solver_state, iteration, stopping, update_pmm
│   ├── Interface/         solver, solver_status
│   └── qp_problem.hpp
├── src/                   (implementations, same grouping)
├── tests/                 (doctest unit tests, same grouping)
├── third_party/           (fetched by setup.sh; gitignored)
├── build/                 (compiler output; gitignored)
├── setup.sh
└── Makefile
```

---

## How it works

**Linear algebra (`LinearAlgebra/`).** CSC `SparseMatrix` with matvec, transpose
matvec, explicit transpose, and 1/∞ norms; `SymSparseMatrix` storing a symmetric
matrix by its upper triangle (the form QDLDL consumes) with a symmetric matvec and
norm; and BLAS-1 vector kernels (`dot`, `axpy`, `norm`, `all_finite`).

**KKT system (`KKT/`).** `KKTSystem` assembles the regularized Newton matrix

```
K = [ -(Q + Θ⁻¹ + ρI)   Aᵀ  ]
    [        A          δI  ]
```

storing its upper triangle. The sparsity **pattern** is built once; each iteration
only refreshes **values** via `assemble(Θ⁻¹, ρ, δ)`. `KKTSolver` wraps QDLDL with a
three-tier lifetime — symbolic analysis once, numeric factorization per iteration,
triangular solves per right-hand side — and `newton_backsolve` forms the reduced
right-hand side and recovers the bound-dual directions.

**IP-PMM control (`IPM_Control/`).** The algorithm layer: two-sided complementarity
and Θ⁻¹, the fraction-to-boundary ratio test, proximal-regularized residuals,
Mehrotra predictor–corrector, boundary control, the Mehrotra warm start, one full
iteration, the optimality stopping test, and the PMM parameter/estimate updates.

**Interface (`Interface/`).** `solve()` initializes the warm-started state, builds
the KKT machinery once, and runs the outer loop (residuals → stopping → iteration →
PMM update) to a `SolveResult`.

---

## Roadmap

- [x] **Phase 0 — Toolchain & environment**
- [x] **Phase 1 — Project scaffolding**
- [x] **Phase 2 — Sparse matrix (CSC)** with matvec, transpose, norms
- [x] **Phase 3 — Vector kernels & testing** (BLAS-1, doctest)
- [x] **Phase 4 — Problem representation & residuals** (two-sided bounds)
- [x] **Phase 5 — Newton/KKT system** (`SymSparseMatrix`, `KKTSystem`, QDLDL wrapper)
- [x] **Phase 6 — IP-PMM iteration** (warm start, Mehrotra predictor–corrector,
      step control, proximal updates, optimality stopping) — **solves QPs end to end**
- [ ] **Robustness hardening** — retry on ill-conditioning, `avoid_local_min`,
      infeasibility detection (primal/dual), the full six-way termination status
- [ ] **Presolve & input** — parse standard QP formats (NETLIB, Maros–Mészáros),
      eliminate fixed variables and other reductions
- [ ] **Validation & performance** — cross-check against the MATLAB reference,
      AMD fill-reducing reordering, profiling

---

## Licensing

Released under the **MIT License** (see `LICENSE`). All dependencies are permissively
licensed and MIT-compatible:

| Component | Role | License |
| --------- | ---- | ------- |
| [doctest](https://github.com/doctest/doctest) | Unit-test framework (dev only) | MIT |
| [QDLDL](https://github.com/osqp/qdldl) | Sparse LDLᵀ factorization of the quasidefinite KKT system | Apache-2.0 |
| [AMD](https://github.com/DrTimothyAldenDavis/SuiteSparse) *(planned)* | Fill-reducing ordering | BSD-3-Clause (dual-licensed) |

Permissive dependencies require attribution, not relicensing: each keeps its own
license, and the combined work remains free to distribute under MIT. GPL/LGPL
components (e.g. CHOLMOD) are intentionally avoided.

---

## Acknowledgements

Based on the IP-PMM method and its reference MATLAB implementations. See:

> S. Pougkakiotis and J. Gondzio, *An Interior Point-Proximal Method of Multipliers
> for Convex Quadratic Programming*, Computational Optimization and Applications,
> 78 (2021), pp. 307–351.