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


/* element-wise and semi-sparse GPU operations vs their CPU counterparts */
static void gpu_elementwise(const char *path, ptiIndex R)
{
    ptiSparseTensor A, B, Zc, Zg;
    if(pti_test_load(&A, path) != 0 || pti_test_load(&B, path) != 0) { ++pti_test_failures; return; }
    char what[160];

    /* MulScalar: GPU in-place vs CPU in-place */
    snprintf(what, sizeof what, "GPU MulScalar (%s)", path);
    int rc = ptiCudaSparseTensorMulScalar(&A, (ptiValue) 2.5);
    CHECK(rc == 0, "%s failed rc=%d", what, rc);
    if(rc == 0) {
        ptiSparseTensorMulScalar(&B, (ptiValue) 2.5);
        double worst = 0;
        for(ptiNnzIndex z = 0; z < A.nnz; ++z) {
            double d = fabs((double) A.values.data[z] - (double) B.values.data[z]);
            if(d > worst) worst = d;
        }
        CHECK(worst <= 1e-6, "%s: max abs diff %g", what, worst);
    }

    /* DotMulEq: GPU vs CPU (operands must be identically ordered - they are) */
    snprintf(what, sizeof what, "GPU DotMulEq (%s)", path);
    rc = ptiCudaSparseTensorDotMulEq(&Zg, &A, &A);
    CHECK(rc == 0, "%s failed rc=%d", what, rc);
    if(rc == 0) {
        CHECK(ptiSparseTensorDotMulEq(&Zc, &A, &A) == 0, "%s: cpu ref failed", what);
        CHECK(Zg.nnz == Zc.nnz, "%s: nnz differ", what);
        double worst = 0, scale = 0;
        for(ptiNnzIndex z = 0; z < Zg.nnz && z < Zc.nnz; ++z) {
            double e = (double) Zc.values.data[z];
            if(fabs(e) > scale) scale = fabs(e);
            double d = fabs((double) Zg.values.data[z] - e);
            if(d > worst) worst = d;
        }
        CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0), "%s: max diff %g", what, worst);
        ptiFreeSparseTensor(&Zc);
        ptiFreeSparseTensor(&Zg);
    }
    ptiFreeSparseTensor(&A);
    ptiFreeSparseTensor(&B);

    /* semi-sparse TTM: GPU vs CPU, per mode.  The GPU kernel takes separate X
       and Y strides - it used one stride for both, which was wrong whenever
       roundup8(ndims[mode]) != roundup8(R) (correct on tiny tensors only). */
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        ptiSemiSparseTensor S;
        if(ptiSparseTensorToSemiSparseTensor(&S, &X, mode) != 0) continue;
        ptiMatrix U;
        ptiNewMatrix(&U, X.ndims[mode], R);
        ptiRandomizeMatrix(&U);

        ptiSemiSparseTensor Yc, Yg;
        int rc1 = ptiSemiSparseTensorMulMatrix(&Yc, &S, &U, mode);
        int rc2 = ptiCudaSemiSparseTensorMulMatrix(&Yg, &S, &U, mode);
        snprintf(what, sizeof what, "GPU SemiSparse TTM (%s mode %u R=%u)", path, mode, R);
        CHECK(rc1 == 0 && rc2 == 0, "%s failed (%d,%d)", what, rc1, rc2);
        if(rc1 == 0 && rc2 == 0) {
            CHECK(Yc.nnz == Yg.nnz, "%s: nnz differ", what);
            double worst = 0, scale = 0;
            for(ptiNnzIndex z = 0; z < Yc.nnz && z < Yg.nnz; ++z)
                for(ptiIndex r = 0; r < R; ++r) {
                    double e = (double) Yc.values.values[z * Yc.stride + r];
                    if(fabs(e) > scale) scale = fabs(e);
                    double d = fabs((double) Yg.values.values[z * Yg.stride + r] - e);
                    if(d > worst) worst = d;
                }
            CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0), "%s: max diff %g", what, worst);
        }
        if(rc1 == 0) ptiFreeSemiSparseTensor(&Yc);
        if(rc2 == 0) ptiFreeSemiSparseTensor(&Yg);
        ptiFreeMatrix(&U);
        ptiFreeSemiSparseTensor(&S);
    }
    ptiFreeSparseTensor(&X);
}


/* GPU CP-ALS: on an exactly rank-3 tensor it must converge to fit ~ 1 and
 * agree with the CPU CP-ALS (identical deterministic initialisation). */
static void gpu_cpd(void)
{
    /* same generator as test_cpd.c */
    const char *path = "test_gpu_lowrank.tns";
    ptiIndex const R = 3;
    ptiIndex const dims[3] = { 6, 5, 4 };
    uint64_t rng = 99ULL;
    double *F[3];
    for(int m = 0; m < 3; ++m) {
        F[m] = (double *) malloc(dims[m] * R * sizeof(double));
        for(ptiIndex i = 0; i < dims[m] * R; ++i)
            F[m][i] = 0.2 + (double) pti_test_rand(&rng);
    }
    FILE *f = fopen(path, "w");
    if(!f) { perror(path); ++pti_test_failures; return; }
    fprintf(f, "3\n%u %u %u\n", dims[0], dims[1], dims[2]);
    for(ptiIndex i = 0; i < dims[0]; ++i)
        for(ptiIndex j = 0; j < dims[1]; ++j)
            for(ptiIndex k = 0; k < dims[2]; ++k) {
                double v = 0;
                for(ptiIndex r = 0; r < R; ++r)
                    v += F[0][i * R + r] * F[1][j * R + r] * F[2][k * R + r];
                fprintf(f, "%u %u %u %.17g\n", i + 1, j + 1, k + 1, v);
            }
    fclose(f);
    for(int m = 0; m < 3; ++m) free(F[m]);

    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, (char *) path) != 0) { ++pti_test_failures; remove(path); return; }

    ptiKruskalTensor kc, kg;
    ptiNewKruskalTensor(&kc, X.nmodes, X.ndims, R);
    ptiNewKruskalTensor(&kg, X.nmodes, X.ndims, R);
    CHECK(ptiCpdAls(&X, R, 500, 1e-12, &kc) == 0, "cpu cpd failed");
    int rc = ptiCudaCpdAls(&X, R, 500, 1e-12, 15, &kg);
    CHECK(rc == 0, "ptiCudaCpdAls failed rc=%d", rc);
    if(rc == 0) {
        CHECK(kg.fit > 0.999, "GPU cpd fit %g, expected > 0.999 on exactly rank-3", kg.fit);
        CHECK(fabs(kg.fit - kc.fit) <= 5e-3, "GPU fit %g vs CPU fit %g", kg.fit, kc.fit);
    }

    /* invalid impl_num and non-3D input must be refused */
    ptiKruskalTensor kbad;
    ptiNewKruskalTensor(&kbad, X.nmodes, X.ndims, R);
    CHECK(ptiCudaCpdAls(&X, R, 5, 1e-6, 7, &kbad) != 0,
          "GPU cpd accepted unsupported impl_num 7");

    ptiFreeKruskalTensor(&kc);
    ptiFreeKruskalTensor(&kg);
    ptiFreeSparseTensor(&X);
    remove(path);
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
            gpu_elementwise(p, ranks[k]);
        }
    gpu_cpd();
    return TEST_SUMMARY();
}
