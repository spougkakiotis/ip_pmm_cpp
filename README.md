# IP-PMM (C++)

A C++ implementation of an **Interior Point–Proximal Method of Multipliers (IP-PMM)**
for convex quadratic programming, supporting general two-sided bounds. This is a
from-scratch port of the reference MATLAB solvers by Spyridon Pougkakiotis
([IP_PMM](https://github.com/spougkakiotis/IP_PMM) and the double-bound
[IP-PMM_QP_Solver](https://github.com/spougkakiotis/IP-PMM_QP_Solver)), built around
sparse linear algebra and a permissively licensed sparse factorization.

The solver targets problems of the form

```
minimize     cᵀx + ½ xᵀ Q x
subject to   A x = b
             lb ≤ x ≤ ub
```

where `A` is `m × n`, `Q` is symmetric positive semidefinite (stored upper-triangular),
and each bound may be finite or infinite. Setting `lb_i = -∞` and `ub_i = +∞` makes
variable `i` free; a one-sided bound uses an infinite value on the unconstrained side.
This two-sided formulation subsumes the classic `x ≥ 0` standard form as a special case.

> **Status: under active development.** The linear-algebra foundation and the KKT
> system — sparse matrices, the symmetric factorization, and the assembled Newton
> system — are complete and tested. The interior-point iteration itself (Phase 6)
> is next. This does **not** yet solve QPs end to end.

---

## Design goals

- **Fully permissive licensing.** MIT-licensed, depending only on permissively
  licensed components (see [Licensing](#licensing)); no GPL/LGPL dependencies.
- **Sparse throughout.** Problem data and the Newton systems are stored and factored
  in compressed sparse column (CSC) form — the same layout MATLAB uses internally,
  and the format the factorization backend consumes directly.
- **Quasidefinite by construction.** IP-PMM's primal–dual proximal regularization
  makes each Newton (KKT) system quasidefinite, which admits a pivot-free LDLᵀ
  factorization — matched exactly by the QDLDL backend.
- **Built bottom-up and tested.** Every kernel has unit tests with hand-computed
  oracles; higher layers are composed from tested primitives.

---

## Requirements

- A C++20 compiler (developed with `g++` 13)
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
via CMake. Everything it fetches lives under `third_party/`, which is gitignored —
so the dependencies are reconstructed from `setup.sh` rather than committed.

### Make targets

| Target       | Effect                                              |
| ------------ | --------------------------------------------------- |
| `make`       | Build the solver executable (`build/ip_pmm`)        |
| `make run`   | Build, then run the solver                          |
| `make test`  | Build and run the full unit-test suite              |
| `make clean` | Remove all build artifacts (`build/`)               |

The build uses `-std=c++20 -Wall -Wextra -Wpedantic` and currently compiles at
`-O0 -g` for debuggable development builds. Header dependencies are tracked
automatically (`-MMD -MP`). The vendored QDLDL (C) is compiled with `gcc`;
everything else is C++.

---

## Project layout

Sources are grouped by module. Headers live under `include/`, implementations under
`src/`, and unit tests under `tests/`, with matching subfolders.

```
ip_pmm_cpp/
├── include/
│   ├── LinearAlgebra/
│   │   ├── sparse_matrix.hpp
│   │   ├── sym_sparse_matrix.hpp
│   │   └── vector_ops.hpp
│   ├── KKT/
│   │   ├── kkt_system.hpp
│   │   └── kkt_solver.hpp
│   ├── IPM_Control/
│   │   └── residuals.hpp
│   └── qp_problem.hpp
├── src/
│   ├── LinearAlgebra/   (sparse_matrix, sym_sparse_matrix, vector_ops)
│   ├── KKT/             (kkt_system, kkt_solver)
│   ├── IPM_Control/     (residuals)
│   ├── qp_problem.cpp
│   └── main.cpp         (solver driver — placeholder for now)
├── tests/               (doctest unit tests, mirroring the module groups)
├── third_party/         (fetched by setup.sh; gitignored)
├── build/               (compiler output; gitignored)
├── setup.sh
└── Makefile
```

---

## What's implemented

**Linear algebra (`LinearAlgebra/`)**
- `SparseMatrix` — CSC storage with `A x`, `Aᵀ x` (matrix-free), and an explicit
  counting-sort `transpose()`.
- `SymSparseMatrix` — a symmetric matrix stored upper-triangular (composition over
  `SparseMatrix`), with a symmetric matvec. This is the form QDLDL consumes directly,
  so it doubles as the currency between assembly and the factorizer.
- `vector_ops` — `dot`, in-place `axpy`, `norm` (1/2/∞ via a `Norm` enum), and
  `all_finite` for detecting non-finite iterates after a solve.

**Problem representation (`qp_problem.hpp`)**
- `QPProblem` holds `c`, `A`, `Q`, `b`, and the bound vectors `lb`, `ub`
  (with `±∞` sentinels). Bound regimes (free / lower-only / upper-only / boxed)
  are derived via `has_lower`, `has_upper`, `is_free` — no separate stored mask.
- `validate()` throws `std::invalid_argument` on dimension mismatch or crossed bounds.

**KKT system (`KKT/`)**
- `KKTSystem` assembles the regularized Newton matrix

  ```
  K = [ -(Q + Θ⁻¹ + ρI)   Aᵀ  ]
      [        A          δI  ]
  ```

  storing its upper triangle. The sparsity **pattern** is computed once (constructor);
  each iteration only refreshes the **values** via `assemble(Θ⁻¹, ρ, δ)` — matching
  the symbolic-once factorization design. A diagonal-merge step reserves the (1,1)
  diagonal even when `Q` lacks it, so `Q` stays minimally populated (LP ⇒ empty `Q`).
- `KKTSolver` — an RAII wrapper around QDLDL with a three-tier lifetime: symbolic
  analysis once (constructor), numeric factorization per iteration (`factorize`),
  and triangular solves per right-hand side (`solve`). Reusing one factorization for
  multiple solves is exactly what Mehrotra's predictor–corrector needs.

**Residuals (`IPM_Control/`)**
- Primal infeasibility `r_p = A x − b`.
- Dual infeasibility `r_d = c + Q x − Aᵀ y − z_l + z_u`, with separate multipliers
  for the lower and upper bounds.

All of the above are covered by the unit-test suite, including end-to-end
assemble → factorize → solve on quasidefinite systems.

---

## Roadmap

- [x] **Phase 0 — Toolchain & environment** (compiler, VS Code + WSL, git)
- [x] **Phase 1 — Project scaffolding** (layout, Makefile, `.gitignore`, GitHub)
- [x] **Phase 2 — Sparse matrix (CSC)** with `A x`, `Aᵀ x`, transpose
- [x] **Phase 3 — Vector kernels & testing** (BLAS-1 ops, `norm`, `all_finite`, doctest)
- [x] **Phase 4 — Problem representation & residuals** (two-sided bounds, `r_p`, `r_d`)
- [x] **Phase 5 — Newton/KKT system** (`SymSparseMatrix`, `KKTSystem`, QDLDL wrapper)
- [ ] **Phase 6 — IP-PMM iteration** (Θ⁻¹ and two-sided complementarity, Mehrotra
      predictor–corrector, step control, proximal `ρ`/`δ` updates, stopping criteria)
- [ ] **Phase 7 — Validation & benchmarking** (NETLIB/Maros–Mészáros problems,
      cross-check against the MATLAB reference, profiling)

---

## Licensing

Released under the **MIT License** (see `LICENSE`). All dependencies are permissively
licensed and MIT-compatible:

| Component | Role | License |
| --------- | ---- | ------- |
| [doctest](https://github.com/doctest/doctest) | Unit-test framework (dev only) | MIT |
| [QDLDL](https://github.com/osqp/qdldl) | Sparse LDLᵀ factorization of the quasidefinite KKT system | Apache-2.0 |
| [AMD](https://github.com/DrTimothyAldenDavis/SuiteSparse) *(optional, planned)* | Fill-reducing ordering | BSD-3-Clause (dual-licensed) |

Permissive dependencies require attribution, not relicensing: each keeps its own
license, and the combined work remains free to distribute under MIT. GPL/LGPL
components (e.g. CHOLMOD) are intentionally avoided.

---

## Acknowledgements

Based on the IP-PMM method and its reference MATLAB implementations by
Spyridon Pougkakiotis. See:

> S. Pougkakiotis and J. Gondzio, *An Interior Point-Proximal Method of Multipliers
> for Convex Quadratic Programming*, Computational Optimization and Applications,
> 78 (2021), pp. 307–351.