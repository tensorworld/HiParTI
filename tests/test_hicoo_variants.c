/* The HiCOO MTTKRP MatrixTiling family (ptiRankMatrix-based), sequential and
 * the OpenMP Scheduled/Reduce variants - 695 lines of mttkrp_omp.c that until
 * now had essentially no coverage.  Everything is compared against the naive
 * COO oracle on the same (truly random) factors.
 */
#include "test_util.h"

static const char *DATA_DIR;

/* naive MTTKRP reading factors out of ptiRankMatrix structs */
static void ref_mttkrp_rank(const ptiSparseTensor *X, ptiRankMatrix **U,
                            ptiIndex mode, double *out, ptiIndex R)
{
    memset(out, 0, (size_t) X->ndims[mode] * R * sizeof *out);
    for(ptiNnzIndex z = 0; z < X->nnz; ++z) {
        ptiIndex const row = X->inds[mode].data[z];
        for(ptiIndex r = 0; r < R; ++r) {
            double p = (double) X->values.data[z];
            for(ptiIndex m = 0; m < X->nmodes; ++m) {
                if(m == mode) continue;
                p *= (double) U[m]->values[X->inds[m].data[z] * U[m]->stride + r];
            }
            out[(size_t) row * R + r] += p;
        }
    }
}

static void check_out(const ptiRankMatrix *got, const double *ref,
                      ptiIndex nrows, ptiIndex R, const char *what)
{
    double scale = 0, worst = 0, gotmax = 0;
    for(ptiIndex i = 0; i < nrows; ++i)
        for(ptiIndex r = 0; r < R; ++r) {
            double e = ref[(size_t) i * R + r];
            double g = (double) got->values[i * got->stride + r];
            if(fabs(e) > scale) scale = fabs(e);
            if(fabs(g) > gotmax) gotmax = fabs(g);
            double d = fabs(g - e);
            if(d > worst) worst = d;
        }
    CHECK(!(scale > 0 && gotmax == 0), "%s: output untouched (all zeros)", what);
    CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0),
          "%s: max abs diff %g (scale %g)", what, worst, scale);
}

static void run_matrixtiling(const char *path, ptiIndex R,
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

    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        ptiIndex const nmodes = X.nmodes;
        ptiIndex max_ndims = 0;
        ptiRankMatrix **U = (ptiRankMatrix **) malloc((nmodes + 1) * sizeof *U);
        for(ptiIndex m = 0; m < nmodes; ++m) {
            U[m] = (ptiRankMatrix *) malloc(sizeof(ptiRankMatrix));
            ptiNewRankMatrix(U[m], X.ndims[m], (ptiElementIndex) R);
            ptiRandomizeRankMatrix(U[m], X.ndims[m], (ptiElementIndex) R);
            if(X.ndims[m] > max_ndims) max_ndims = X.ndims[m];
        }
        U[nmodes] = (ptiRankMatrix *) malloc(sizeof(ptiRankMatrix));
        ptiNewRankMatrix(U[nmodes], max_ndims, (ptiElementIndex) R);
        ptiConstantRankMatrix(U[nmodes], 0);

        ptiIndex *order = (ptiIndex *) malloc(nmodes * sizeof *order);
        order[0] = mode;
        for(ptiIndex i = 1, m = (mode + 1) % nmodes; i < nmodes; ++i, m = (m + 1) % nmodes)
            order[i] = m;

        double *ref = (double *) malloc((size_t) X.ndims[mode] * R * sizeof *ref);
        ref_mttkrp_rank(&X, U, mode, ref, R);

        char what[192];

        /* sequential matrix-tiled */
        ptiConstantRankMatrix(U[nmodes], 0);
        snprintf(what, sizeof what, "HiCOO MatrixTiling seq (%s mode %u R=%u)", path, mode, R);
        CHECK(ptiMTTKRPHiCOO_MatrixTiling(&H, U, order, mode) == 0, "%s failed", what);
        check_out(U[nmodes], ref, X.ndims[mode], R, what);

#ifdef HIPARTI_USE_OPENMP
        int const tks[] = { 2, 4 };
        for(size_t t = 0; t < sizeof(tks)/sizeof(*tks); ++t) {
            int const tk = tks[t];

            ptiConstantRankMatrix(U[nmodes], 0);
            snprintf(what, sizeof what, "MatrixTiling omp tk=%d (%s mode %u R=%u)", tk, path, mode, R);
            if(ptiOmpMTTKRPHiCOO_MatrixTiling(&H, U, order, mode, tk, 1) == 0)
                check_out(U[nmodes], ref, X.ndims[mode], R, what);

            for(int balanced = 0; balanced <= 1; ++balanced) {
                ptiConstantRankMatrix(U[nmodes], 0);
                snprintf(what, sizeof what, "MatrixTiling Scheduled tk=%d bal=%d (%s mode %u R=%u)",
                         tk, balanced, path, mode, R);
                if(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled(&H, U, order, mode, tk, 1, balanced) == 0)
                    check_out(U[nmodes], ref, X.ndims[mode], R, what);
            }

            /* privatised (Reduce) variant */
            ptiRankMatrix **copy_U = (ptiRankMatrix **) malloc(tk * sizeof *copy_U);
            for(int i = 0; i < tk; ++i) {
                copy_U[i] = (ptiRankMatrix *) malloc(sizeof(ptiRankMatrix));
                ptiNewRankMatrix(copy_U[i], X.ndims[mode], (ptiElementIndex) R);
                ptiConstantRankMatrix(copy_U[i], 0);
            }
            ptiConstantRankMatrix(U[nmodes], 0);
            snprintf(what, sizeof what, "MatrixTiling Scheduled_Reduce tk=%d (%s mode %u R=%u)",
                     tk, path, mode, R);
            if(ptiOmpMTTKRPHiCOO_MatrixTiling_Scheduled_Reduce(&H, U, copy_U, order, mode, tk, 1, 0) == 0)
                check_out(U[nmodes], ref, X.ndims[mode], R, what);
            for(int i = 0; i < tk; ++i) { ptiFreeRankMatrix(copy_U[i]); free(copy_U[i]); }
            free(copy_U);
        }
#endif
        free(ref); free(order);
        for(ptiIndex m = 0; m <= nmodes; ++m) { ptiFreeRankMatrix(U[m]); free(U[m]); }
        free(U);
    }
    ptiFreeSparseTensorHiCOO(&H);
    ptiFreeSparseTensor(&X);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    char p[1024];
    snprintf(p, sizeof p, "%s/tensors/3d_7.tns", DATA_DIR);
    run_matrixtiling(p, 4, 2, 4, 2);
    snprintf(p, sizeof p, "%s/tensors/3d-24.tns", DATA_DIR);
    run_matrixtiling(p, 8, 2, 4, 2);
    snprintf(p, sizeof p, "%s/tensors/3D_12031.tns", DATA_DIR);
    run_matrixtiling(p, 16, 4, 6, 4);
    run_matrixtiling(p, 8, 4, 6, 4);
    return TEST_SUMMARY();
}
