/* GPU correctness, built only when USE_CUDA=ON.
 *
 * Every CUDA kernel variant is checked against the same CPU oracle the other
 * tests use.  This is the coverage that was missing entirely: the GPU paths
 * were exercised only by benchmarks, which check the exit code and print a
 * GFLOP/s number but never look at the result - so a kernel that launched with
 * a zero-sized grid and computed nothing still "passed".
 */
#include "test_util.h"

static const char *DATA_DIR;

static ptiMatrix ** build_factors(const ptiSparseTensor *X, ptiIndex R, ptiIndex *max_ndims_out)
{
    ptiIndex const nmodes = X->nmodes;
    ptiMatrix **U = (ptiMatrix **) malloc((nmodes + 1) * sizeof *U);
    ptiIndex max_ndims = 0;
    for(ptiIndex m = 0; m < nmodes; ++m) {
        U[m] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
        ptiNewMatrix(U[m], X->ndims[m], R);
        ptiRandomizeMatrix(U[m]);
        if(X->ndims[m] > max_ndims) max_ndims = X->ndims[m];
    }
    U[nmodes] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
    ptiNewMatrix(U[nmodes], max_ndims, R);
    ptiConstantMatrix(U[nmodes], 0);
    *max_ndims_out = max_ndims;
    return U;
}

static void release(ptiMatrix **U, ptiIndex nmodes)
{
    for(ptiIndex m = 0; m <= nmodes; ++m) { ptiFreeMatrix(U[m]); free(U[m]); }
    free(U);
}

static void compare(const ptiMatrix *got, const ptiMatrix *ref, ptiIndex nrows, const char *what)
{
    double scale = pti_test_max_abs_matrix(ref), worst = 0;
    for(ptiIndex i = 0; i < nrows; ++i)
        for(ptiIndex r = 0; r < ref->ncols; ++r) {
            double d = fabs((double) got->values[i * got->stride + r]
                          - (double) ref->values[i * ref->stride + r]);
            if(d > worst) worst = d;
        }
    /* A kernel that never ran leaves the output at zero; make that unmistakable. */
    double gotmax = 0;
    for(ptiIndex i = 0; i < nrows; ++i)
        for(ptiIndex r = 0; r < ref->ncols; ++r) {
            double v = fabs((double) got->values[i * got->stride + r]);
            if(v > gotmax) gotmax = v;
        }
    CHECK(!(scale > 0 && gotmax == 0), "%s: result is all zeros (kernel did not run?)", what);
    CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0),
          "%s: max abs diff %g (scale %g)", what, worst, scale);
}

static void coo_gpu(const char *path, ptiIndex R)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    if(X.nmodes != 3) { ptiFreeSparseTensor(&X); return; }   /* CUDA kernels are 3-D only */

    int const impls[] = { 11, 15, 16 };
    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        ptiIndex max_ndims;
        ptiMatrix **U = build_factors(&X, R, &max_ndims);
        ptiIndex *order = (ptiIndex *) malloc(X.nmodes * sizeof *order);
        order[0] = mode;
        for(ptiIndex i = 1, m = (mode + 1) % X.nmodes; i < X.nmodes; ++i, m = (m + 1) % X.nmodes)
            order[i] = m;

        ptiMatrix ref;
        ptiNewMatrix(&ref, X.ndims[mode], R);
        pti_test_ref_mttkrp(&X, U, mode, &ref);

        for(size_t k = 0; k < sizeof(impls)/sizeof(*impls); ++k) {
            ptiConstantMatrix(U[X.nmodes], 0);
            int rc = ptiCudaMTTKRPOneKernel(&X, U, order, mode, impls[k]);
            char what[192];
            snprintf(what, sizeof what, "COO GPU impl=%d (mode %u R=%u)", impls[k], mode, R);
            CHECK(rc == 0, "%s: returned %d", what, rc);
            if(rc == 0) compare(U[X.nmodes], &ref, X.ndims[mode], what);
        }

        /* an unsupported impl_num must be refused, never silently ignored */
        ptiConstantMatrix(U[X.nmodes], 0);
        CHECK(ptiCudaMTTKRPOneKernel(&X, U, order, mode, 3) != 0,
              "COO GPU accepted unsupported impl_num 3");

        ptiFreeMatrix(&ref); free(order);
        release(U, X.nmodes);
    }
    ptiFreeSparseTensor(&X);
}

static void hicoo_gpu(const char *path, ptiIndex R,
                      ptiElementIndex sb, ptiElementIndex sk, ptiElementIndex sc)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    if(X.nmodes != 3) { ptiFreeSparseTensor(&X); return; }

    ptiSparseTensorHiCOO H;
    ptiNnzIndex max_nnzb = 0;
    if(ptiSparseTensorToHiCOO(&H, &max_nnzb, &X, sb, sk, sc, 1) != 0) {
        ptiFreeSparseTensor(&X); return;
    }

    int const impls[] = { 1, 2, 3, 4, 14, 15, 16 };
    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        ptiIndex max_ndims;
        ptiMatrix **U = build_factors(&X, R, &max_ndims);
        ptiIndex *order = (ptiIndex *) malloc(X.nmodes * sizeof *order);
        order[0] = mode;
        for(ptiIndex i = 1, m = (mode + 1) % X.nmodes; i < X.nmodes; ++i, m = (m + 1) % X.nmodes)
            order[i] = m;

        ptiMatrix ref;
        ptiNewMatrix(&ref, X.ndims[mode], R);
        pti_test_ref_mttkrp(&X, U, mode, &ref);

        for(size_t k = 0; k < sizeof(impls)/sizeof(*impls); ++k) {
            ptiConstantMatrix(U[X.nmodes], 0);
            int rc = ptiCudaMTTKRPHiCOO(&H, U, order, mode, max_nnzb, impls[k]);
            char what[192];
            snprintf(what, sizeof what, "HiCOO GPU impl=%d (mode %u R=%u b=%u)",
                     impls[k], mode, R, (unsigned) sb);
            /* Some variants legitimately refuse a configuration (e.g. more than
               1024 threads per block); a refusal is fine, a wrong answer is not. */
            if(rc == 0) compare(U[X.nmodes], &ref, X.ndims[mode], what);
        }

        ptiConstantMatrix(U[X.nmodes], 0);
        CHECK(ptiCudaMTTKRPHiCOO(&H, U, order, mode, max_nnzb, 7) != 0,
              "HiCOO GPU accepted unsupported impl_num 7");

        ptiFreeMatrix(&ref); free(order);
        release(U, X.nmodes);
    }
    ptiFreeSparseTensorHiCOO(&H);
    ptiFreeSparseTensor(&X);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    if(ptiCudaSetDevice(0) != 0) {
        fprintf(stderr, "no usable CUDA device; skipping\n");
        return 77;              /* CTest treats 77 as "skipped" */
    }

    char p[1024];
    const char *tensors[] = { "3d_7.tns", "3d-24.tns", "3D_12031.tns" };
    const ptiIndex ranks[] = { 8, 16, 32 };
    for(size_t t = 0; t < sizeof(tensors)/sizeof(*tensors); ++t)
        for(size_t k = 0; k < sizeof(ranks)/sizeof(*ranks); ++k) {
            snprintf(p, sizeof p, "%s/tensors/%s", DATA_DIR, tensors[t]);
            coo_gpu(p, ranks[k]);
            hicoo_gpu(p, ranks[k], 2, 4, 2);
        }
    return TEST_SUMMARY();
}
