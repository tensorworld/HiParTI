/* Cross-implementation equivalence.
 *
 * Every implementation of an operation must agree with every other one.  This
 * needs no hand-computed expected values, and it is the check that catches the
 * failure mode this codebase turned out to be prone to: one code path quietly
 * computing nothing (or something wrong) while still returning success.
 *
 * Covered here: COO MTTKRP (sequential / OpenMP / OpenMP+privatisation, over a
 * range of thread counts) and HiCOO MTTKRP (sequential / OpenMP).
 */
#include "test_util.h"

static const char *DATA_DIR;

typedef struct {
    ptiMatrix **U;
    ptiIndex   *order;
    ptiIndex    nmodes;
    ptiIndex    max_ndims;
} factors_t;

static factors_t make_factors(const ptiSparseTensor *X, ptiIndex R, ptiIndex mode)
{
    factors_t f;
    f.nmodes = X->nmodes;
    f.U = (ptiMatrix **) malloc((f.nmodes + 1) * sizeof *f.U);
    f.max_ndims = 0;
    for(ptiIndex m = 0; m < f.nmodes; ++m) {
        f.U[m] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
        ptiNewMatrix(f.U[m], X->ndims[m], R);
        ptiRandomizeMatrix(f.U[m]);
        if(X->ndims[m] > f.max_ndims) f.max_ndims = X->ndims[m];
    }
    f.U[f.nmodes] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
    ptiNewMatrix(f.U[f.nmodes], f.max_ndims, R);
    ptiConstantMatrix(f.U[f.nmodes], 0);

    f.order = (ptiIndex *) malloc(f.nmodes * sizeof *f.order);
    f.order[0] = mode;
    for(ptiIndex i = 1, m = (mode + 1) % f.nmodes; i < f.nmodes; ++i, m = (m + 1) % f.nmodes)
        f.order[i] = m;
    return f;
}

static void free_factors(factors_t *f)
{
    for(ptiIndex m = 0; m <= f->nmodes; ++m) { ptiFreeMatrix(f->U[m]); free(f->U[m]); }
    free(f->U); free(f->order);
}

/* snapshot the first `nrows` rows of the output matrix */
static ptiValue * snapshot(const ptiMatrix *M, ptiIndex nrows)
{
    ptiValue *buf = (ptiValue *) malloc((size_t) nrows * M->ncols * sizeof *buf);
    for(ptiIndex i = 0; i < nrows; ++i)
        for(ptiIndex r = 0; r < M->ncols; ++r)
            buf[(size_t) i * M->ncols + r] = M->values[i * M->stride + r];
    return buf;
}

static void agree(const ptiValue *a, const ptiValue *b, ptiIndex nrows, ptiIndex R,
                  const char *what)
{
    double scale = 0, worst = 0;
    for(size_t i = 0; i < (size_t) nrows * R; ++i) {
        double x = fabs((double) b[i]);
        if(x > scale) scale = x;
        double d = fabs((double) a[i] - (double) b[i]);
        if(d > worst) worst = d;
    }
    CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0),
          "%s: max abs diff %g (scale %g)", what, worst, scale);
}

static void coo_equivalence(const char *path, ptiIndex R)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }

    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        factors_t f = make_factors(&X, R, mode);
        ptiIndex const nrows = X.ndims[mode];
        char what[192];

        /* the oracle */
        ptiMatrix ref;
        ptiNewMatrix(&ref, nrows, R);
        pti_test_ref_mttkrp(&X, f.U, mode, &ref);
        ptiValue *vref = snapshot(&ref, nrows);

        /* sequential */
        ptiConstantMatrix(f.U[f.nmodes], 0);
        CHECK(ptiMTTKRP(&X, f.U, f.order, mode) == 0, "ptiMTTKRP failed");
        ptiValue *vseq = snapshot(f.U[f.nmodes], nrows);
        snprintf(what, sizeof what, "seq vs reference (%s mode %u R=%u)", path, mode, R);
        agree(vseq, vref, nrows, R, what);

#ifdef HIPARTI_USE_OPENMP
        int const threads[] = { 1, 2, 4, 8 };
        for(size_t t = 0; t < sizeof(threads)/sizeof(*threads); ++t) {
            ptiConstantMatrix(f.U[f.nmodes], 0);
            if(ptiOmpMTTKRP(&X, f.U, f.order, mode, threads[t]) == 0) {
                ptiValue *v = snapshot(f.U[f.nmodes], nrows);
                snprintf(what, sizeof what, "omp(tk=%d) vs seq (%s mode %u R=%u)",
                         threads[t], path, mode, R);
                agree(v, vseq, nrows, R, what);
                free(v);
            }

            /* privatised variant: needs one scratch matrix per thread */
            ptiMatrix **copy_U = (ptiMatrix **) malloc(threads[t] * sizeof *copy_U);
            for(int i = 0; i < threads[t]; ++i) {
                copy_U[i] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
                ptiNewMatrix(copy_U[i], nrows, R);
                ptiConstantMatrix(copy_U[i], 0);
            }
            ptiConstantMatrix(f.U[f.nmodes], 0);
            if(ptiOmpMTTKRP_Reduce(&X, f.U, copy_U, f.order, mode, threads[t]) == 0) {
                ptiValue *v = snapshot(f.U[f.nmodes], nrows);
                snprintf(what, sizeof what, "omp_reduce(tk=%d) vs seq (%s mode %u R=%u)",
                         threads[t], path, mode, R);
                agree(v, vseq, nrows, R, what);
                free(v);
            }
            for(int i = 0; i < threads[t]; ++i) { ptiFreeMatrix(copy_U[i]); free(copy_U[i]); }
            free(copy_U);
        }
#endif
        free(vseq); free(vref);
        ptiFreeMatrix(&ref);
        free_factors(&f);
    }
    ptiFreeSparseTensor(&X);
}

static void hicoo_equivalence(const char *path, ptiIndex R,
                              ptiElementIndex sb, ptiElementIndex sk, ptiElementIndex sc)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    if(X.nmodes != 3) { ptiFreeSparseTensor(&X); return; }  /* HiCOO kernels are 3-D only */

    ptiSparseTensorHiCOO H;
    ptiNnzIndex max_nnzb = 0;
    if(ptiSparseTensorToHiCOO(&H, &max_nnzb, &X, sb, sk, sc, 1) != 0) {
        ptiFreeSparseTensor(&X); return;   /* config rejected (e.g. too many nnz per block) */
    }

    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        factors_t f = make_factors(&X, R, mode);
        ptiIndex const nrows = X.ndims[mode];
        char what[192];

        ptiMatrix ref;
        ptiNewMatrix(&ref, nrows, R);
        pti_test_ref_mttkrp(&X, f.U, mode, &ref);
        ptiValue *vref = snapshot(&ref, nrows);

        ptiConstantMatrix(f.U[f.nmodes], 0);
        CHECK(ptiMTTKRPHiCOO(&H, f.U, f.order, mode) == 0, "ptiMTTKRPHiCOO failed");
        ptiValue *vhc = snapshot(f.U[f.nmodes], nrows);
        snprintf(what, sizeof what, "HiCOO seq vs reference (%s mode %u R=%u b=%u)",
                 path, mode, R, (unsigned) sb);
        agree(vhc, vref, nrows, R, what);

#ifdef HIPARTI_USE_OPENMP
        /* tk==1 && tb==1 is explicitly not supported by the library */
        int const threads[] = { 2, 4, 8 };
        for(size_t t = 0; t < sizeof(threads)/sizeof(*threads); ++t) {
            ptiConstantMatrix(f.U[f.nmodes], 0);
            if(ptiOmpMTTKRPHiCOO(&H, f.U, f.order, mode, threads[t], 1) == 0) {
                ptiValue *v = snapshot(f.U[f.nmodes], nrows);
                snprintf(what, sizeof what, "HiCOO omp(tk=%d) vs HiCOO seq (%s mode %u R=%u)",
                         threads[t], path, mode, R);
                agree(v, vhc, nrows, R, what);
                free(v);
            }
        }
#endif
        free(vhc); free(vref);
        ptiFreeMatrix(&ref);
        free_factors(&f);
    }
    ptiFreeSparseTensorHiCOO(&H);
    ptiFreeSparseTensor(&X);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    char p[1024];
    const char *tensors[] = { "3d_7.tns", "3d-24.tns", "3D_12031.tns", "4d_3_16.tns" };
    const ptiIndex ranks[] = { 4, 16 };

    for(size_t t = 0; t < sizeof(tensors)/sizeof(*tensors); ++t) {
        snprintf(p, sizeof p, "%s/tensors/%s", DATA_DIR, tensors[t]);
        for(size_t k = 0; k < sizeof(ranks)/sizeof(*ranks); ++k) {
            coo_equivalence(p, ranks[k]);
            hicoo_equivalence(p, ranks[k], 2, 4, 2);
        }
    }
    return TEST_SUMMARY();
}
