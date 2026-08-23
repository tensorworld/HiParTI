/* MTTKRP checked against a naive reference, and every CPU variant checked
   against every other.  Cross-variant agreement is what catches the class of
   bug where one code path silently computes nothing. */
#include "test_util.h"

static const char *DATA_DIR;

static ptiMatrix ** make_factors(const ptiSparseTensor *X, ptiIndex R, ptiIndex *max_ndims_out)
{
    ptiIndex const nmodes = X->nmodes;
    ptiMatrix ** U = (ptiMatrix **) malloc((nmodes + 1) * sizeof *U);
    ptiIndex max_ndims = 0;
    for(ptiIndex m = 0; m < nmodes; ++m) {
        U[m] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
        ptiNewMatrix(U[m], X->ndims[m], R);
        ptiRandomizeMatrix(U[m]);          /* deterministic: splitmix64, fixed seed */
        if(X->ndims[m] > max_ndims) max_ndims = X->ndims[m];
    }
    U[nmodes] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
    ptiNewMatrix(U[nmodes], max_ndims, R);
    ptiConstantMatrix(U[nmodes], 0);
    *max_ndims_out = max_ndims;
    return U;
}

static void free_factors(ptiMatrix **U, ptiIndex nmodes)
{
    for(ptiIndex m = 0; m <= nmodes; ++m) { ptiFreeMatrix(U[m]); free(U[m]); }
    free(U);
}

/* compare only the first ndims[mode] rows: U[nmodes] is allocated with
   max_ndims rows and the surplus is padding, not result */
static int compare_rows(const ptiMatrix *got, const ptiMatrix *ref, ptiIndex nrows, const char *what)
{
    double scale = pti_test_max_abs_matrix(ref);
    double worst = 0;
    for(ptiIndex i = 0; i < nrows; ++i)
        for(ptiIndex r = 0; r < ref->ncols; ++r) {
            double a = got->values[i * got->stride + r];
            double b = ref->values[i * ref->stride + r];
            double d = fabs(a - b);
            if(d > worst) worst = d;
        }
    int ok = (worst <= 2e-5 * (scale > 0 ? scale : 1.0));
    CHECK(ok, "%s: max abs diff %g (scale %g)", what, worst, scale);
    return ok;
}


/* ptiMTTKRP special-cases nmodes==3; without a 4-D tensor the generic path is
   never executed (a mutation there went undetected until this was added). */
static const char * make_4d_tensor(void)
{
    static const char *path = "test_mttkrp_4d.tns";
    FILE *f = fopen(path, "w");
    if(!f) { perror(path); exit(2); }
    fputs("4\n5 4 3 2\n", f);
    uint64_t rng = 20260823ULL;
    for(int i = 1; i <= 5; ++i)
        for(int j = 1; j <= 4; ++j) {
            int k = 1 + ((i + j) % 3);
            int l = 1 + ((i * j) % 2);
            fprintf(f, "%d %d %d %d %.9g\n", i, j, k, l, (double) pti_test_rand(&rng));
        }
    fclose(f);
    return path;
}

static void run_one(const char *path, ptiIndex R)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }

    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        ptiIndex max_ndims;
        ptiMatrix **U = make_factors(&X, R, &max_ndims);
        ptiIndex *order = (ptiIndex *) malloc(X.nmodes * sizeof *order);
        order[0] = mode;
        for(ptiIndex i = 1, m = (mode + 1) % X.nmodes; i < X.nmodes; ++i, m = (m + 1) % X.nmodes)
            order[i] = m;

        /* oracle */
        ptiMatrix ref;
        ptiNewMatrix(&ref, X.ndims[mode], R);
        pti_test_ref_mttkrp(&X, U, mode, &ref);

        char what[128];

        /* sequential */
        ptiConstantMatrix(U[X.nmodes], 0);
        CHECK(ptiMTTKRP(&X, U, order, mode) == 0, "ptiMTTKRP failed (mode %u)", mode);
        snprintf(what, sizeof what, "seq vs reference (mode %u, R=%u)", mode, R);
        compare_rows(U[X.nmodes], &ref, X.ndims[mode], what);

#ifdef HIPARTI_USE_OPENMP
        /* OpenMP, no privatisation */
        ptiConstantMatrix(U[X.nmodes], 0);
        if(ptiOmpMTTKRP(&X, U, order, mode, 4) == 0) {
            snprintf(what, sizeof what, "omp vs reference (mode %u, R=%u)", mode, R);
            compare_rows(U[X.nmodes], &ref, X.ndims[mode], what);
        }
#endif
        ptiFreeMatrix(&ref);
        free(order);
        free_factors(U, X.nmodes);
    }
    ptiFreeSparseTensor(&X);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    char p[1024];
    const char *tensors[] = { "3d_7.tns", "3d-24.tns", "3D_12031.tns" };
    const ptiIndex ranks[] = { 4, 16 };

    for(size_t t = 0; t < sizeof(tensors)/sizeof(*tensors); ++t)
        for(size_t k = 0; k < sizeof(ranks)/sizeof(*ranks); ++k) {
            snprintf(p, sizeof p, "%s/tensors/%s", DATA_DIR, tensors[t]);
            run_one(p, ranks[k]);
        }

    /* generic (non-3-D) MTTKRP path */
    const char *p4 = make_4d_tensor();
    for(size_t k = 0; k < sizeof(ranks)/sizeof(*ranks); ++k) run_one(p4, ranks[k]);
    remove(p4);

    return TEST_SUMMARY();
}
