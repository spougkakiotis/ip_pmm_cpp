#!/usr/bin/env bash
# Fetch third-party dependencies into third_party/ (gitignored).
# Run once after cloning:  ./setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

# ---- doctest (unit-test framework, header-only, MIT) ----
echo "Fetching doctest..."
mkdir -p "$TP/doctest"
curl -fsSL -o "$TP/doctest/doctest.h" \
  https://raw.githubusercontent.com/doctest/doctest/master/doctest/doctest.h

# ---- QDLDL (sparse LDL^T factorization, Apache-2.0) ----
QDLDL_TAG="v0.1.7"
if [ ! -d "$TP/qdldl" ]; then
  echo "Cloning QDLDL $QDLDL_TAG..."
  git clone --depth 1 --branch "$QDLDL_TAG" \
    https://github.com/osqp/qdldl.git "$TP/qdldl"
else
  echo "QDLDL already present, skipping clone."
fi

# ---- Hand-written qdldl_types.h (replaces the CMake-generated one) ----
echo "Writing qdldl_types.h..."
cat > "$TP/qdldl/include/qdldl_types.h" <<'EOF'
#ifndef QDLDL_TYPES_H
#define QDLDL_TYPES_H

#include <limits.h>

typedef int           QDLDL_int;
typedef double        QDLDL_float;
typedef unsigned char QDLDL_bool;

#define QDLDL_INT_MAX INT_MAX

#endif /* QDLDL_TYPES_H */
EOF

# ---- Hand-written qdldl_version.h (also CMake-generated upstream) ----
echo "Writing qdldl_version.h..."
cat > "$TP/qdldl/include/qdldl_version.h" <<'EOF'
#ifndef QDLDL_VERSION_H
#define QDLDL_VERSION_H

#define QDLDL_VERSION_MAJOR 0
#define QDLDL_VERSION_MINOR 1
#define QDLDL_VERSION_PATCH 7

#endif /* QDLDL_VERSION_H */
EOF

echo "Done. Dependencies are in third_party/ (gitignored)."
