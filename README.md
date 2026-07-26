# IP-PMM (C++)

A C++ implementation of an **Interior Point–Proximal Method of Multipliers (IP-PMM)**
for convex quadratic programming. This is a from-scratch port of the reference
MATLAB solver ([spougkakiotis/IP_PMM](https://github.com/spougkakiotis/IP_PMM)),
built around sparse linear algebra and a permissively licensed sparse factorization.

The solver targets problems of the form

```
minimize     cᵀx + ½ xᵀ Q x
subject to   A x = b
             x_i ≥ 0   for constrained variables
             x_i free  otherwise
```

where `A` is `m × n`, `Q` is symmetric positive semidefinite, and a subset of the
variables may be free (unbounded).

> **Status: under active development.** The numerical foundations (sparse matrix
> type, vector kernels, problem representation, residuals) are complete and tested.
> The interior-point iteration and linear-system solve are in progress — see the
> [Roadmap](#roadmap). This does **not** yet solve QPs end to end.

---

## Design goals

- **Fully permissive licensing.** The project is MIT-licensed and depends only on
  permissively licensed components (see [Licensing](#licensing)). No GPL/LGPL
  dependencies.
- **Sparse throughout.** Problem data and the Newton systems are stored and
  factored in compressed sparse column (CSC) form — the same layout MATLAB uses
  internally, and the format expected by the factorization backend.
- **Built bottom-up and tested.** Every numerical kernel has unit tests with
  hand-computed oracles; higher layers are composed from tested primitives.

---

## Requirements

- A C++20 compiler (developed with `g++` 13)
- GNU `make`
- `curl` (to fetch the test framework, once)

Developed on Ubuntu (native or via WSL2 on Windows). Any Unix-like toolchain works.

---

## Getting started

Clone, fetch the (header-only) test framework, then build:

```bash
git clone git@github.com:spougkakiotis/ip_pmm_cpp.git
cd ip_pmm_cpp

# Fetch doctest into third_party/ (gitignored, not vendored in history)
mkdir -p third_party/doctest
curl -L -o third_party/doctest/doctest.h \
  https://raw.githubusercontent.com/doctest/doctest/master/doctest/doctest.h

make          # build the solver binary   -> build/ip_pmm
make test     # build and run the test suite
```

### Make targets

| Target       | Effect                                              |
| ------------ | --------------------------------------------------- |
| `make`       | Build the solver executable (`build/ip_pmm`)        |
| `make run`   | Build, then run the solver                          |
| `make test`  | Build and run the full unit-test suite              |
| `make clean` | Remove all build artifacts (`build/`)               |

The build uses `-std=c++20 -Wall -Wextra -Wpedantic` and currently compiles at
`-O0 -g` for debuggable development builds. Header dependencies are tracked
automatically (`-MMD -MP`), so editing a header rebuilds exactly what depends on it.

---

## Project layout

```
ip_pmm_cpp/
├── include/            # public headers (declarations)
│   ├── sparse_matrix.hpp
│   ├── vector_ops.hpp
│   ├── qp_problem.hpp
│   └── residuals.hpp
├── src/                # implementations
│   ├── main.cpp        # entry point (solver driver — placeholder for now)
│   ├── sparse_matrix.cpp
│   ├── vector_ops.cpp
│   ├── qp_problem.cpp
│   └── residuals.cpp
├── tests/              # doctest unit tests (one file per module)
│   ├── test_main.cpp   # defines the test runner's main()
│   ├── test_sparse_matrix.cpp
│   ├── test_vector_ops.cpp
│   └── test_residuals.cpp
├── third_party/        # fetched dependencies (gitignored)
├── build/              # compiler output (gitignored)
└── Makefile
```

---

## What's implemented

**Sparse matrix (`SparseMatrix`, CSC)**
- Three-array compressed sparse column storage (`col_ptr`, `row_idx`, `values`)
- Matrix–vector product `y = A x`
- Transpose product `y = Aᵀ x` (computed from the same arrays, no explicit transpose)
- Invariant checks via `assert` on construction

**Dense vector kernels (`vector_ops`)**
- `dot(x, y)` — inner product
- `axpy(α, x, y)` — in-place `y ← αx + y`
- `norm(x, type)` — 1-, 2-, and ∞-norms via a `Norm` enum selector (2-norm default)
- `all_finite(x)` — guard for detecting non-finite iterates after a linear solve

**Problem representation (`QPProblem`)**
- Holds `c`, `A`, `Q`, `b`, and a per-variable free/bounded mask
- Mirrors the reference MATLAB interface `IP_PMM(c, A, Q, b, free_variables)`
- `validate()` throws `std::invalid_argument` on dimension mismatch

**Residuals (`residuals`)**
- Primal infeasibility `r_p = A x − b`
- Dual infeasibility `r_d = c + Q x − Aᵀ y − z`

All of the above are covered by the unit-test suite.

---

## Roadmap

Checkpoints reflect development progress.

- [x] **Phase 0 — Toolchain & environment** (compiler, VS Code + WSL, git, hello world)
- [x] **Phase 1 — Project scaffolding** (layout, Makefile, `.gitignore`, GitHub)
- [x] **Phase 2 — Sparse matrix (CSC)** with `A x` and `Aᵀ x`
- [x] **Phase 3 — Vector kernels & testing** (BLAS-1 ops, `norm`, `all_finite`, doctest harness)
- [x] **Phase 4 — Problem representation & residuals** (`QPProblem`, validation, `r_p`, `r_d`)
- [ ] **Phase 5 — Newton/KKT system** (quasidefinite augmented system, QDLDL factorization)
- [ ] **Phase 6 — IP-PMM iteration** (proximal regularization, predictor–corrector, step control, stopping criteria)
- [ ] **Phase 7 — Validation & benchmarking** (NETLIB problems, cross-check against the MATLAB reference, profiling)

---

## Licensing

This project is released under the **MIT License** (see `LICENSE`).

Planned and current dependencies are all permissively licensed and MIT-compatible:

| Component | Role | License |
| --------- | ---- | ------- |
| [doctest](https://github.com/doctest/doctest) | Unit-test framework (dev only) | MIT |
| [QDLDL](https://github.com/osqp/qdldl) *(planned)* | Sparse LDLᵀ factorization of the quasidefinite KKT system | Apache-2.0 |
| [AMD](https://github.com/DrTimothyAldenDavis/SuiteSparse) *(optional, planned)* | Fill-reducing ordering | BSD-3-Clause (dual-licensed) |

Permissive dependencies require attribution, not relicensing: each retains its own
license, and the combined work remains free to distribute under MIT. GPL/LGPL
components (e.g. CHOLMOD) are intentionally avoided.

---

## Acknowledgements

Based on the IP-PMM method and its reference MATLAB implementation by
Spyridon Pougkakiotis. See:

> S. Pougkakiotis and J. Gondzio, *An Interior Point-Proximal Method of Multipliers
> for Convex Quadratic Programming*, Computational Optimization and Applications,
> 78 (2021), pp. 307–351.