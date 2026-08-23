hiparti - Python interface
==========================

A small, dependency-free (standard library only, via `ctypes`) Python wrapper
for the HiParTI C library.  It replaces the former cffi-based `HiParTIPy`.

Setup
-----

Nothing to install.  Build the library first (`./build.sh`), then either run
Python from the repository root, or point the wrapper at the shared library:

```sh
export HIPARTI_LIB=/path/to/libHiParTI.so     # optional
```

If the library was built with `-DHIPARTI_VALUE_TYPEWIDTH=64`, also set
`HIPARTI_VALUE_TYPEWIDTH=64` so both sides agree on the value type.

Usage
-----

```python
import sys; sys.path.insert(0, "python")   # when running from the repo root
import hiparti

X = hiparti.SparseTensor.load("data/tensors/3d_7.tns")
print(X)                      # SparseTensor(ndims=[4, 4, 3], nnz=9)
print(X.ndims, X.nnz, X.sum())
for coords, value in X.entries():
    print(coords, value)

X.mul_scalar(2.5)             # in place
X.dump("out.tns")

Z = X.add(X)                  # element-wise: add/sub/dot_mul/dot_div
M = X.mttkrp(mode=0, rank=8)  # -> DenseMatrix (deterministic random factors)
M2 = X.mttkrp(mode=0, rank=8, threads=4)     # OpenMP variant

fit, lam, factors = X.cpd(rank=8, niters=50, tol=1e-6)

A = hiparti.SparseMatrix.load("data/matrices/tiny.mtx")
y = A.spmv([1.0] * A.ncols)   # SpMV -> list
C = A.spmm(rank=8)            # SpMM -> DenseMatrix
```

Testing
-------

```sh
python3 python/test_hiparti.py            # from the repository root
```

The same script runs under `ctest` when python3 is available.

Notes
-----

* Tensor files may be native `.tns` or TensorSuite format (auto-detected).
* `DenseMatrix(..., init="random")` uses the library's deterministic
  splitmix64 generator, so runs are reproducible.
* The wrapper checks every return code and raises `RuntimeError` on failure;
  file handles are opened and closed properly.
