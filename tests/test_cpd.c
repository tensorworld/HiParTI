/* CP decomposition (CP-ALS).
 *
 * Factor matrices are only defined up to permutation and scaling, so exact
 * values cannot be pinned down.  What can be checked:
 *   - the reported fit is in [0, 1] and not degenerate
 *   - reconstructing the tensor from (lambda, factors) approximates the input,
 *     and for an exactly rank-R input recovers it to high accuracy
 *   - sequential and OpenMP agree on the fit for the same input
 */
#include "test_util.h"

static const char *DATA_DIR;

/* || X - [[lambda; U1..UN]] ||^2 evaluated at the nonzeros of X plus the norm of
   the reconstruction restricted elsewhere is expensive; instead evaluate the
   reconstruction at X's nonzeros and compare pointwise. For an exactly low-rank
   dense-ish small tensor this is a strong check. */
static double recon_at(const ptiKruskalTensor *k, const ptiIndex *idx)
{
    double s = 0;
    for(ptiIndex r = 0; r < k->rank; ++r) {
        double p = (double) k->lambda[r];
        for(ptiIndex m = 0; m < k->nmodes; ++m)
            p *= (double) k->factors[m]->values[idx[m] * k->factors[m]->stride + r];
        s += p;
    }
    return s;
}

/* build a small dense tensor that is exactly rank `R` from random factors */
static void make_lowrank_tensor(const char *path, ptiIndex R)
{
    ptiIndex const dims[3] = { 6, 5, 4 };
    uint64_t rng = 99ULL;
    double *U[3];
    for(int m = 0; m < 3; ++m) {
        U[m] = (double *) malloc(dims[m] * R * sizeof(double));
        for(ptiIndex i = 0; i < dims[m] * R; ++i)
            U[m][i] = 0.2 + (double) pti_test_rand(&rng);   /* positive, well-conditioned */
    }
    FILE *f = fopen(path, "w");
    if(!f) { perror(path); exit(2); }
    fprintf(f, "3\n%u %u %u\n", dims[0], dims[1], dims[2]);
    for(ptiIndex i = 0; i < dims[0]; ++i)
        for(ptiIndex j = 0; j < dims[1]; ++j)
            for(ptiIndex k = 0; k < dims[2]; ++k) {
                double v = 0;
                for(ptiIndex r = 0; r < R; ++r)
                    v += U[0][i * R + r] * U[1][j * R + r] * U[2][k * R + r];
                fprintf(f, "%u %u %u %.17g\n", i + 1, j + 1, k + 1, v);
            }
    fclose(f);
    for(int m = 0; m < 3; ++m) free(U[m]);
}

static double run_cpd_seq(const ptiSparseTensor *X, ptiIndex R, const char *what)
{
    ptiKruskalTensor kt;
    ptiNewKruskalTensor(&kt, X->nmodes, X->ndims, R);
    int rc = ptiCpdAls(X, R, 500, 1e-12, &kt);   /* ALS converges slowly on collinear factors */
    CHECK(rc == 0, "%s: ptiCpdAls failed rc=%d", what, rc);
    double fit = kt.fit;
    CHECK(fit > 0.0 && fit <= 1.0 + 1e-9, "%s: fit %g outside (0,1]", what, fit);

    /* reconstruction error at the nonzeros */
    if(rc == 0) {
        double scale = 0, worst = 0;
        ptiIndex idx[8];
        for(ptiNnzIndex z = 0; z < X->nnz; ++z) {
            for(ptiIndex m = 0; m < X->nmodes; ++m) idx[m] = X->inds[m].data[z];
            double v = (double) X->values.data[z];
            if(fabs(v) > scale) scale = fabs(v);
            double d = fabs(recon_at(&kt, idx) - v);
            if(d > worst) worst = d;
        }
        /* generous: ALS on a general tensor only approximates; tightened below
           for the exactly low-rank input via the fit instead */
        CHECK(worst <= 2.0 * scale + 1e-9,
              "%s: reconstruction wildly off (max err %g, scale %g)", what, worst, scale);
    }
    /* the Kruskal dump routine must run and produce output */
    if(rc == 0) {
        FILE *f = tmpfile();
        if(f) {
            CHECK(ptiDumpKruskalTensor(&kt, f) == 0, "%s: DumpKruskalTensor failed", what);
            CHECK(ftell(f) > 0, "%s: DumpKruskalTensor wrote nothing", what);
            fclose(f);
        }
    }
    ptiFreeKruskalTensor(&kt);
    return fit;
}

static void test_lowrank_recovery(void)
{
    const char *p = "test_cpd_lowrank.tns";
    ptiIndex const R = 3;
    make_lowrank_tensor(p, R);

    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, (char *) p) != 0) { ++pti_test_failures; remove(p); return; }

    double fit = run_cpd_seq(&X, R, "cpd exact-rank");
    /* the input IS rank 3, so rank-3 ALS should essentially nail it */
    CHECK(fit > 0.999, "cpd exact-rank: fit %g, expected > 0.999 on an exactly rank-3 tensor", fit);

    /* more capacity must never fit worse */
    double fit_r5 = run_cpd_seq(&X, 5, "cpd overcomplete");
    CHECK(fit_r5 >= fit - 5e-3, "fit decreased with larger rank: %g -> %g", fit, fit_r5);

#ifdef HIPARTI_USE_OPENMP
    ptiKruskalTensor kt;
    ptiNewKruskalTensor(&kt, X.nmodes, X.ndims, R);
    int rc = ptiOmpCpdAls(&X, R, 500, 1e-12, 4, 1, &kt);
    CHECK(rc == 0, "ptiOmpCpdAls failed rc=%d", rc);
    if(rc == 0)
        CHECK(fabs(kt.fit - fit) <= 5e-3,
              "omp fit %g differs from seq fit %g", kt.fit, fit);
    ptiFreeKruskalTensor(&kt);
#endif

    ptiFreeSparseTensor(&X);
    remove(p);
}

static void test_on_data_tensor(void)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/tensors/3d_7.tns", DATA_DIR);
    ptiSparseTensor X;
    if(pti_test_load(&X, p) != 0) { ++pti_test_failures; return; }
    run_cpd_seq(&X, 4, "cpd 3d_7");
    ptiFreeSparseTensor(&X);
}


/* HiCOO CP-ALS goes through the ptiRankKruskalTensor fit path */
static void test_hicoo_cpd(void)
{
    const char *p = "test_cpd_lowrank_h.tns";
    ptiIndex const R = 3;
    make_lowrank_tensor(p, R);

    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, (char *) p) != 0) { ++pti_test_failures; remove(p); return; }

    ptiSparseTensorHiCOO H;
    ptiNnzIndex max_nnzb = 0;
    if(ptiSparseTensorToHiCOO(&H, &max_nnzb, &X, 2, 4, 2, 1) == 0) {
        ptiRankKruskalTensor kt;
        ptiNewRankKruskalTensor(&kt, X.nmodes, X.ndims, (ptiElementIndex) R);
        int rc = ptiCpdAlsHiCOO(&H, R, 500, 1e-12, &kt);
        CHECK(rc == 0, "ptiCpdAlsHiCOO failed rc=%d", rc);
        if(rc == 0) {
            CHECK(kt.fit > 0.0 && kt.fit <= 1.0 + 1e-9, "HiCOO cpd fit %g outside (0,1]", kt.fit);
            CHECK(kt.fit > 0.999, "HiCOO cpd fit %g, expected > 0.999 on exactly rank-3", kt.fit);
        }
        ptiFreeRankKruskalTensor(&kt);

#ifdef HIPARTI_USE_OPENMP
        ptiRankKruskalTensor kto;
        ptiNewRankKruskalTensor(&kto, X.nmodes, X.ndims, (ptiElementIndex) R);
        rc = ptiOmpCpdAlsHiCOO(&H, R, 500, 1e-12, 4, 1, 0, &kto);
        CHECK(rc == 0, "ptiOmpCpdAlsHiCOO failed rc=%d", rc);
        if(rc == 0)
            CHECK(kto.fit > 0.999, "HiCOO omp cpd fit %g, expected > 0.999", kto.fit);
        ptiFreeRankKruskalTensor(&kto);
#endif
        ptiFreeSparseTensorHiCOO(&H);
    } else {
        CHECK(0, "COO->HiCOO failed for the CPD test tensor");
    }
    ptiFreeSparseTensor(&X);
    remove(p);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    test_lowrank_recovery();
    test_hicoo_cpd();
    test_on_data_tensor();
    return TEST_SUMMARY();
}
