"""hiparti - a small, dependency-free Python interface to HiParTI.

Uses only the standard library (ctypes), so it works on machines without
pip.  Point it at the shared library with the HIPARTI_LIB environment
variable, or run from the repository root after ./build.sh (it looks in
./build), or install the library system-wide.

    import hiparti
    X = hiparti.SparseTensor.load("data/tensors/3d_7.tns")
    print(X.nmodes, X.ndims, X.nnz)

    M = X.mttkrp(mode=0, rank=8)          # -> DenseMatrix
    fit, lam, factors = X.cpd(rank=8)     # CP decomposition

    A = hiparti.SparseMatrix.load("data/matrices/tiny.mtx")
    y = A.spmv([1.0] * A.ncols)

If the library was built with -DHIPARTI_VALUE_TYPEWIDTH=64, set
HIPARTI_VALUE_TYPEWIDTH=64 in the environment so both sides agree on the
value type.
"""

import ctypes
import ctypes.util
import os

__all__ = ["SparseTensor", "SparseMatrix", "DenseMatrix", "load_library"]

# ---------------------------------------------------------------------------
# scalar types - must match the build (include/includes/types.h)
# ---------------------------------------------------------------------------
_VALUE_WIDTH = int(os.environ.get("HIPARTI_VALUE_TYPEWIDTH", "32"))
_INDEX_WIDTH = int(os.environ.get("HIPARTI_INDEX_TYPEWIDTH", "32"))

Value = ctypes.c_double if _VALUE_WIDTH == 64 else ctypes.c_float
Index = ctypes.c_uint64 if _INDEX_WIDTH == 64 else ctypes.c_uint32
NnzIndex = ctypes.c_uint64
ElementIndex = ctypes.c_uint8


# ---------------------------------------------------------------------------
# struct layouts - must mirror include/includes/structs.h
# ---------------------------------------------------------------------------
class _ValueVector(ctypes.Structure):
    _fields_ = [("len", NnzIndex), ("cap", NnzIndex), ("data", ctypes.POINTER(Value))]

class _IndexVector(ctypes.Structure):
    _fields_ = [("len", NnzIndex), ("cap", NnzIndex), ("data", ctypes.POINTER(Index))]

class _Matrix(ctypes.Structure):
    _fields_ = [("nrows", Index), ("ncols", Index), ("cap", Index),
                ("stride", Index), ("values", ctypes.POINTER(Value))]

class _SparseTensor(ctypes.Structure):
    _fields_ = [("nmodes", Index),
                ("sortorder", ctypes.POINTER(Index)),
                ("ndims", ctypes.POINTER(Index)),
                ("nnz", NnzIndex),
                ("inds", ctypes.POINTER(_IndexVector)),
                ("values", _ValueVector)]

class _SparseMatrix(ctypes.Structure):
    _fields_ = [("nrows", Index), ("ncols", Index), ("nnz", NnzIndex),
                ("rowind", _IndexVector), ("colind", _IndexVector),
                ("values", _ValueVector)]

class _KruskalTensor(ctypes.Structure):
    _fields_ = [("nmodes", Index), ("rank", Index),
                ("ndims", ctypes.POINTER(Index)),
                ("lambda_", ctypes.POINTER(Value)),
                ("fit", ctypes.c_double),
                ("factors", ctypes.POINTER(ctypes.POINTER(_Matrix)))]


# ---------------------------------------------------------------------------
# library loading
# ---------------------------------------------------------------------------
_lib = None

def load_library(path=None):
    """Load libHiParTI and declare the wrapped prototypes.  Called
    automatically on first use; call explicitly to point at a specific
    build."""
    global _lib
    if path is None:
        candidates = [os.environ.get("HIPARTI_LIB"),
                      os.path.join("build", "libHiParTI.so"),
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "build", "libHiParTI.so"),
                      ctypes.util.find_library("HiParTI")]
        path = next((c for c in candidates if c and os.path.exists(c)), None)
        if path is None and candidates[-1]:
            path = candidates[-1]          # a system library needs no exists()
    if path is None:
        raise OSError("cannot find libHiParTI.so; set HIPARTI_LIB or run "
                      "from the repository root after ./build.sh")
    _lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    P = ctypes.POINTER
    proto = {
        # tensor basics
        "ptiLoadSparseTensor":   (ctypes.c_int, [P(_SparseTensor), Index, ctypes.c_char_p]),
        "ptiDumpSparseTensor":   (ctypes.c_int, [P(_SparseTensor), Index, ctypes.c_void_p]),
        "ptiFreeSparseTensor":   (None,         [P(_SparseTensor)]),
        "ptiCopySparseTensor":   (ctypes.c_int, [P(_SparseTensor), P(_SparseTensor), ctypes.c_int]),
        "ptiSparseTensorSortIndex": (None,      [P(_SparseTensor), ctypes.c_int, ctypes.c_int]),
        "ptiSparseTensorMulScalar": (ctypes.c_int, [P(_SparseTensor), Value]),
        "ptiSparseTensorDivScalar": (ctypes.c_int, [P(_SparseTensor), Value]),
        "ptiSparseTensorAdd":    (ctypes.c_int, [P(_SparseTensor), P(_SparseTensor), P(_SparseTensor)]),
        "ptiSparseTensorSub":    (ctypes.c_int, [P(_SparseTensor), P(_SparseTensor), P(_SparseTensor)]),
        "ptiSparseTensorDotMul": (ctypes.c_int, [P(_SparseTensor), P(_SparseTensor), P(_SparseTensor)]),
        "ptiSparseTensorDotDiv": (ctypes.c_int, [P(_SparseTensor), P(_SparseTensor), P(_SparseTensor)]),
        # dense matrices
        "ptiNewMatrix":          (ctypes.c_int, [P(_Matrix), Index, Index]),
        "ptiRandomizeMatrix":    (ctypes.c_int, [P(_Matrix)]),
        "ptiConstantMatrix":     (ctypes.c_int, [P(_Matrix), Value]),
        "ptiFreeMatrix":         (None,         [P(_Matrix)]),
        # MTTKRP + CPD
        "ptiMTTKRP":             (ctypes.c_int, [P(_SparseTensor), P(P(_Matrix)), P(Index), Index]),
        "ptiOmpMTTKRP":          (ctypes.c_int, [P(_SparseTensor), P(P(_Matrix)), P(Index), Index, ctypes.c_int]),
        "ptiNewKruskalTensor":   (ctypes.c_int, [P(_KruskalTensor), Index, P(Index), Index]),
        "ptiFreeKruskalTensor":  (None,         [P(_KruskalTensor)]),
        "ptiCpdAls":             (ctypes.c_int, [P(_SparseTensor), Index, Index, ctypes.c_double, P(_KruskalTensor)]),
        # sparse matrices
        "ptiLoadSparseMatrix":   (ctypes.c_int, [P(_SparseMatrix), Index, ctypes.c_void_p]),
        "ptiDumpSparseMatrix":   (ctypes.c_int, [P(_SparseMatrix), Index, ctypes.c_void_p]),
        "ptiFreeSparseMatrix":   (None,         [P(_SparseMatrix)]),
        "ptiSparseMatrixMulVector": (ctypes.c_int, [P(_ValueVector), P(_SparseMatrix), P(_ValueVector)]),
        "ptiSparseMatrixMulMatrix": (ctypes.c_int, [P(_Matrix), P(_SparseMatrix), P(_Matrix)]),
        # value vectors
        "ptiNewValueVector":     (ctypes.c_int, [P(_ValueVector), ctypes.c_uint64, ctypes.c_uint64]),
        "ptiConstantValueVector": (ctypes.c_int, [P(_ValueVector), Value]),
        "ptiFreeValueVector":    (None,         [P(_ValueVector)]),
    }
    for name, (restype, argtypes) in proto.items():
        fn = getattr(_lib, name)          # fails loudly if the symbol is missing
        fn.restype, fn.argtypes = restype, argtypes

    # libc, for FILE* handling done RIGHT (fopen ... fclose)
    libc = ctypes.CDLL(None)
    libc.fopen.restype = ctypes.c_void_p
    libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libc.fclose.restype = ctypes.c_int
    libc.fclose.argtypes = [ctypes.c_void_p]
    _lib._libc = libc
    return _lib


def _L():
    if _lib is None:
        load_library()
    return _lib


def _check(rc, what):
    if rc != 0:
        raise RuntimeError("%s failed (return code %d)" % (what, rc))


class _CFile(object):
    """fopen/fclose done properly (the old cffi wrapper leaked every handle,
    losing buffered output)."""
    def __init__(self, path, mode):
        self.f = _L()._libc.fopen(path.encode(), mode.encode())
        if not self.f:
            raise OSError("cannot open %s" % path)
    def __enter__(self):
        return self.f
    def __exit__(self, *exc):
        _L()._libc.fclose(self.f)
        return False


# ---------------------------------------------------------------------------
# public classes
# ---------------------------------------------------------------------------
class DenseMatrix(object):
    """Thin owner of a ptiMatrix (row-major, stride-padded)."""
    def __init__(self, nrows, ncols, init="zero"):
        L = _L()
        self._m = _Matrix()
        _check(L.ptiNewMatrix(ctypes.byref(self._m), nrows, ncols), "ptiNewMatrix")
        if init == "zero":
            _check(L.ptiConstantMatrix(ctypes.byref(self._m), 0), "ptiConstantMatrix")
        elif init == "random":   # deterministic splitmix64, reproducible runs
            _check(L.ptiRandomizeMatrix(ctypes.byref(self._m)), "ptiRandomizeMatrix")
        self._owned = True

    @property
    def nrows(self): return int(self._m.nrows)
    @property
    def ncols(self): return int(self._m.ncols)

    def __getitem__(self, ij):
        i, j = ij
        return float(self._m.values[i * self._m.stride + j])

    def tolist(self):
        s = self._m.stride
        return [[float(self._m.values[i * s + j]) for j in range(self.ncols)]
                for i in range(self.nrows)]

    def free(self):
        if getattr(self, "_owned", False):
            _L().ptiFreeMatrix(ctypes.byref(self._m))
            self._owned = False
    def __del__(self):
        try: self.free()
        except Exception: pass


class SparseTensor(object):
    """A COO sparse tensor."""
    def __init__(self):
        self._t = _SparseTensor()
        self._owned = False

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path, start_index=1):
        """Read a .tns file (or a TensorSuite file - detected automatically).
        start_index is the index base used in the file (1 for data/)."""
        self = cls()
        _check(_L().ptiLoadSparseTensor(ctypes.byref(self._t), start_index,
                                        path.encode()), "load %s" % path)
        self._owned = True
        return self

    # -- inspection --------------------------------------------------------
    @property
    def nmodes(self): return int(self._t.nmodes)
    @property
    def nnz(self): return int(self._t.nnz)
    @property
    def ndims(self): return [int(self._t.ndims[m]) for m in range(self.nmodes)]

    def index(self, z):
        """Coordinates of the z-th nonzero (0-based)."""
        return tuple(int(self._t.inds[m].data[z]) for m in range(self.nmodes))
    def value(self, z):
        return float(self._t.values.data[z])
    def entries(self):
        """Iterate (coords, value) over all nonzeros."""
        for z in range(self.nnz):
            yield self.index(z), self.value(z)
    def sum(self):
        return sum(self.value(z) for z in range(self.nnz))

    def __repr__(self):
        return "SparseTensor(ndims=%r, nnz=%d)" % (self.ndims, self.nnz)

    # -- basic operations --------------------------------------------------
    def dump(self, path, start_index=1):
        with _CFile(path, "w") as f:
            _check(_L().ptiDumpSparseTensor(ctypes.byref(self._t), start_index, f),
                   "dump %s" % path)

    def sort(self):
        _L().ptiSparseTensorSortIndex(ctypes.byref(self._t), 1, 1)
        return self

    def mul_scalar(self, a):
        _check(_L().ptiSparseTensorMulScalar(ctypes.byref(self._t), a), "mul_scalar")
        return self
    def div_scalar(self, a):
        _check(_L().ptiSparseTensorDivScalar(ctypes.byref(self._t), a), "div_scalar")
        return self

    def _binary(self, other, cfn, name, sort_first=False):
        if sort_first:                 # the dot operators are merge-based
            self.sort(); other.sort()
        out = SparseTensor()
        _check(cfn(ctypes.byref(out._t), ctypes.byref(self._t),
                   ctypes.byref(other._t)), name)
        out._owned = True
        return out
    def add(self, other):     return self._binary(other, _L().ptiSparseTensorAdd, "add")
    def sub(self, other):     return self._binary(other, _L().ptiSparseTensorSub, "sub")
    def dot_mul(self, other): return self._binary(other, _L().ptiSparseTensorDotMul, "dot_mul", True)
    def dot_div(self, other): return self._binary(other, _L().ptiSparseTensorDotDiv, "dot_div", True)

    # -- kernels -----------------------------------------------------------
    def _factors(self, rank, factors):
        n = self.nmodes
        if factors is None:
            factors = [DenseMatrix(self.ndims[m], rank, init="random") for m in range(n)]
        out = DenseMatrix(max(self.ndims), rank, init="zero")
        arr = (ctypes.POINTER(_Matrix) * (n + 1))()
        for m in range(n):
            arr[m] = ctypes.pointer(factors[m]._m)
        arr[n] = ctypes.pointer(out._m)
        return factors, out, arr

    def mttkrp(self, mode, rank=8, factors=None, threads=None):
        """MTTKRP on `mode`.  factors: optional list of DenseMatrix (one per
        mode, ndims[m] x rank); deterministic random ones are created if
        omitted.  Returns the ndims[mode] x rank result as a DenseMatrix."""
        factors, out, arr = self._factors(rank, factors)
        order = (Index * self.nmodes)()
        order[0] = mode
        for i in range(1, self.nmodes):
            order[i] = (mode + i) % self.nmodes
        if threads:
            _check(_L().ptiOmpMTTKRP(ctypes.byref(self._t), arr, order, mode,
                                     threads), "omp mttkrp")
        else:
            _check(_L().ptiMTTKRP(ctypes.byref(self._t), arr, order, mode),
                   "mttkrp")
        out._nkeep = factors            # keep factor storage alive
        return out

    def cpd(self, rank, niters=50, tol=1e-6):
        """CP-ALS decomposition.  Returns (fit, lambda:list, factors:list of
        row-major lists)."""
        L = _L()
        kt = _KruskalTensor()
        dims = (Index * self.nmodes)(*self.ndims)
        _check(L.ptiNewKruskalTensor(ctypes.byref(kt), self.nmodes, dims, rank),
               "new kruskal")
        _check(L.ptiCpdAls(ctypes.byref(self._t), rank, niters, tol,
                           ctypes.byref(kt)), "cpd")
        fit = float(kt.fit)
        lam = [float(kt.lambda_[r]) for r in range(rank)]
        factors = []
        for m in range(self.nmodes):
            fm = kt.factors[m].contents
            factors.append([[float(fm.values[i * fm.stride + j])
                             for j in range(fm.ncols)] for i in range(fm.nrows)])
        L.ptiFreeKruskalTensor(ctypes.byref(kt))
        return fit, lam, factors

    def free(self):
        if self._owned:
            _L().ptiFreeSparseTensor(ctypes.byref(self._t))
            self._owned = False
    def __del__(self):
        try: self.free()
        except Exception: pass


class SparseMatrix(object):
    """A COO sparse matrix read from a MatrixMarket .mtx file."""
    def __init__(self):
        self._m = _SparseMatrix()
        self._owned = False

    @classmethod
    def load(cls, path, start_index=1):
        self = cls()
        with _CFile(path, "r") as f:
            _check(_L().ptiLoadSparseMatrix(ctypes.byref(self._m), start_index, f),
                   "load %s" % path)
        self._owned = True
        return self

    @property
    def nrows(self): return int(self._m.nrows)
    @property
    def ncols(self): return int(self._m.ncols)
    @property
    def nnz(self): return int(self._m.nnz)
    def __repr__(self):
        return "SparseMatrix(%d x %d, nnz=%d)" % (self.nrows, self.ncols, self.nnz)

    def spmv(self, x):
        """y = A @ x for a Python sequence x of length ncols; returns a list."""
        if len(x) != self.ncols:
            raise ValueError("x has length %d, expected %d" % (len(x), self.ncols))
        L = _L()
        xv, yv = _ValueVector(), _ValueVector()
        _check(L.ptiNewValueVector(ctypes.byref(xv), self.ncols, self.ncols), "new x")
        _check(L.ptiNewValueVector(ctypes.byref(yv), self.nrows, self.nrows), "new y")
        for i, v in enumerate(x):
            xv.data[i] = v
        _check(L.ptiConstantValueVector(ctypes.byref(yv), 0), "zero y")
        _check(L.ptiSparseMatrixMulVector(ctypes.byref(yv), ctypes.byref(self._m),
                                          ctypes.byref(xv)), "spmv")
        y = [float(yv.data[i]) for i in range(self.nrows)]
        L.ptiFreeValueVector(ctypes.byref(xv))
        L.ptiFreeValueVector(ctypes.byref(yv))
        return y

    def spmm(self, B=None, rank=8):
        """C = A @ B.  B: DenseMatrix with ncols rows (deterministic random if
        omitted).  Returns C as a DenseMatrix."""
        if B is None:
            B = DenseMatrix(self.ncols, rank, init="random")
        C = DenseMatrix(self.nrows, B.ncols, init="zero")
        _check(_L().ptiSparseMatrixMulMatrix(ctypes.byref(C._m),
                                             ctypes.byref(self._m),
                                             ctypes.byref(B._m)), "spmm")
        C._nkeep = B
        return C

    def free(self):
        if self._owned:
            _L().ptiFreeSparseMatrix(ctypes.byref(self._m))
            self._owned = False
    def __del__(self):
        try: self.free()
        except Exception: pass
