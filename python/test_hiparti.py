#!/usr/bin/env python3
"""Tests for the hiparti Python interface.  Run from the repository root:

    python3 python/test_hiparti.py [path/to/libHiParTI.so]

Registered with CTest, so `ctest` runs this too when python3 is available.
Uses only the standard library.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 1:
    os.environ["HIPARTI_LIB"] = sys.argv[1]

import hiparti

checks = [0, 0]
def check(cond, msg):
    checks[0] += 1
    if not cond:
        checks[1] += 1
        print("  FAIL: %s" % msg)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TNS = os.path.join(root, "data", "tensors", "3d_7.tns")
MTX = os.path.join(root, "data", "matrices", "tiny.mtx")

# --- load + inspect (3d_7: 4x4x3, 9 nnz, value sum 46) ---------------------
X = hiparti.SparseTensor.load(TNS)
check(X.nmodes == 3, "nmodes %d != 3" % X.nmodes)
check(X.ndims == [4, 4, 3], "ndims %r" % X.ndims)
check(X.nnz == 9, "nnz %d != 9" % X.nnz)
check(abs(X.sum() - 46.0) < 1e-6, "value sum %g != 46" % X.sum())
check(X.index(0) == (0, 0, 0) and abs(X.value(0) - 1.0) < 1e-6, "first entry")

# --- scalar ops ------------------------------------------------------------
X.mul_scalar(2.0)
check(abs(X.sum() - 92.0) < 1e-5, "after *2 sum %g != 92" % X.sum())
X.div_scalar(2.0)
check(abs(X.sum() - 46.0) < 1e-5, "after /2 sum %g != 46" % X.sum())

# --- dump round-trip (this was silently broken in the old wrapper) ---------
out = os.path.join(os.getcwd(), "pytest_roundtrip.tns")
X.dump(out)
Y = hiparti.SparseTensor.load(out)
check(Y.nnz == X.nnz, "round-trip nnz %d != %d" % (Y.nnz, X.nnz))
check(abs(Y.sum() - X.sum()) < 1e-5, "round-trip sum")
check(sorted(Y.entries()) == sorted(X.entries()), "round-trip entries differ")
os.remove(out)

# --- element-wise algebra --------------------------------------------------
Z = X.add(Y)
check(abs(Z.sum() - 2 * X.sum()) < 1e-5, "X+X sum %g" % Z.sum())
W = Z.sub(X)
check(abs(W.sum() - X.sum()) < 1e-5, "(X+X)-X sum %g" % W.sum())
D = X.dot_mul(Y)
check(abs(sum(v * v for _, v in X.entries()) - D.sum()) < 1e-4, "X.*X values")
Q = X.dot_div(Y)
check(all(abs(v - 1.0) < 1e-6 for _, v in Q.entries()), "X./X == 1")

# --- MTTKRP vs a pure-Python reference on the same factors -----------------
rank = 4
factors = [hiparti.DenseMatrix(X.ndims[m], rank, init="random") for m in range(3)]
M = X.mttkrp(mode=0, rank=rank, factors=factors)
ref = [[0.0] * rank for _ in range(X.ndims[0])]
for (i, j, k), v in X.entries():
    for r in range(rank):
        ref[i][r] += v * factors[1][j, r] * factors[2][k, r]
worst = max(abs(M[i, r] - ref[i][r]) for i in range(X.ndims[0]) for r in range(rank))
check(worst < 1e-4, "mttkrp vs reference: max diff %g" % worst)

Momp = X.mttkrp(mode=0, rank=rank, factors=factors, threads=2)
worst = max(abs(Momp[i, r] - ref[i][r]) for i in range(X.ndims[0]) for r in range(rank))
check(worst < 1e-4, "omp mttkrp vs reference: max diff %g" % worst)

# --- CPD -------------------------------------------------------------------
fit, lam, fac = X.cpd(rank=4, niters=50, tol=1e-8)
check(0.0 < fit <= 1.0 + 1e-9, "cpd fit %g outside (0,1]" % fit)
check(len(lam) == 4 and len(fac) == 3, "cpd shapes")
check(len(fac[0]) == 4 and len(fac[0][0]) == 4, "factor 0 shape")

# --- sparse matrix ---------------------------------------------------------
A = hiparti.SparseMatrix.load(MTX)
check((A.nrows, A.ncols, A.nnz) == (4, 4, 9), "tiny.mtx shape %r" % (A,))
y = A.spmv([1.0] * A.ncols)
# reference: sum of each row of tiny.mtx
refy = [0.0] * A.nrows
for z in range(A.nnz):
    refy[A._m.rowind.data[z]] += float(A._m.values.data[z])
check(max(abs(a - b) for a, b in zip(y, refy)) < 1e-5, "spmv vs row sums")

B = hiparti.DenseMatrix(A.ncols, 3, init="random")
C = A.spmm(B)
refc = [[0.0] * 3 for _ in range(A.nrows)]
for z in range(A.nnz):
    i, j = A._m.rowind.data[z], A._m.colind.data[z]
    for r in range(3):
        refc[i][r] += float(A._m.values.data[z]) * B[j, r]
worst = max(abs(C[i, r] - refc[i][r]) for i in range(A.nrows) for r in range(3))
check(worst < 1e-4, "spmm vs reference: max diff %g" % worst)

print("%s: %d/%d checks passed"
      % ("FAILED" if checks[1] else "ok", checks[0] - checks[1], checks[0]))
sys.exit(1 if checks[1] else 0)
