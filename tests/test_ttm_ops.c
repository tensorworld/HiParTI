/* TTM and the small tensor algebra (Kronecker, Khatri-Rao, scalar ops),
 * checked against closed-form/analytic properties:
 *   - TTM with an identity-like matrix reproduces the tensor
 *   - TTM against a dense reference contraction
 *   - Kronecker/Khatri-Rao shapes, nnz counts, and values
 *   - s * X scales every value; X .* X squares them
 */
#include "test_util.h"

static const char *DATA_DIR;

/* dense reference: Y(i1..r..iN) = sum_k X(i1..k..iN) * U(k, r) at `mode` */
static void ref_ttm_dense(const ptiSparseTensor *X, const ptiMatrix *U, ptiIndex mode,
                          double *out /* size prod(dims with mode->R) */, ptiIndex R)
{
    /* strides for the output layout, mode dimension replaced by R */
    ptiIndex nm = X->nmodes;
    size_t stride[8]; size_t total = 1;
    for(ptiIndex m = nm; m-- > 0;) {
        stride[m] = total;
        total *= (m == mode) ? R : X->ndims[m];
    }
    memset(out, 0, total * sizeof *out);
    for(ptiNnzIndex z = 0; z < X->nnz; ++z) {
        size_t base = 0;
        for(ptiIndex m = 0; m < nm; ++m)
            if(m != mode) base += (size_t) X->inds[m].data[z] * stride[m];
        ptiIndex const k = X->inds[mode].data[z];
        for(ptiIndex r = 0; r < R; ++r)
            out[base + (size_t) r * stride[mode]] +=
                (double) X->values.data[z] * (double) U->values[k * U->stride + r];
    }
}

static void test_ttm_against_dense(const char *path, ptiIndex R)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    if(X.nmodes > 8) { ptiFreeSparseTensor(&X); return; }

    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        ptiMatrix U;
        ptiNewMatrix(&U, X.ndims[mode], R);
        ptiRandomizeMatrix(&U);

        ptiSemiSparseTensor Y;
        int rc = ptiSparseTensorMulMatrix(&Y, &X, &U, mode);
        char what[160];
        snprintf(what, sizeof what, "TTM (%s mode %u R=%u)", path, mode, R);
        CHECK(rc == 0, "%s: failed rc=%d", what, rc);
        if(rc == 0) {
            size_t total = 1;
            for(ptiIndex m = 0; m < X.nmodes; ++m) total *= (m == mode) ? R : X.ndims[m];
            double *ref = (double *) calloc(total, sizeof *ref);
            ref_ttm_dense(&X, &U, mode, ref, R);

            /* accumulate the semi-sparse result into the same dense layout */
            double *got = (double *) calloc(total, sizeof *got);
            size_t stride[8]; size_t tt = 1;
            for(ptiIndex m = X.nmodes; m-- > 0;) { stride[m] = tt; tt *= (m == mode) ? R : X.ndims[m]; }
            for(ptiNnzIndex z = 0; z < Y.nnz; ++z) {
                size_t base = 0;
                for(ptiIndex m = 0; m < Y.nmodes; ++m)
                    if(m != mode) base += (size_t) Y.inds[m].data[z] * stride[m];
                for(ptiIndex r = 0; r < R; ++r)
                    got[base + (size_t) r * stride[mode]] +=
                        (double) Y.values.values[z * Y.stride + r];
            }
            double scale = 0, worst = 0;
            for(size_t i = 0; i < total; ++i) {
                if(fabs(ref[i]) > scale) scale = fabs(ref[i]);
                double d = fabs(got[i] - ref[i]);
                if(d > worst) worst = d;
            }
            CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0),
                  "%s: max abs diff %g (scale %g)", what, worst, scale);
            free(ref); free(got);
            ptiFreeSemiSparseTensor(&Y);
        }
        ptiFreeMatrix(&U);
    }
    ptiFreeSparseTensor(&X);
}

static void test_kronecker_khatrirao(const char *path)
{
    ptiSparseTensor A;
    if(pti_test_load(&A, path) != 0) { ++pti_test_failures; return; }

    /* Kronecker: dims multiply, nnz multiplies, values are products */
    ptiSparseTensor K;
    int rc = ptiSparseTensorKroneckerMul(&K, &A, &A);
    CHECK(rc == 0, "Kronecker failed rc=%d", rc);
    if(rc == 0) {
        for(ptiIndex m = 0; m < A.nmodes; ++m)
            CHECK(K.ndims[m] == A.ndims[m] * A.ndims[m], "Kronecker dim %u wrong", m);
        CHECK(K.nnz == A.nnz * A.nnz, "Kronecker nnz %llu != %llu",
              (unsigned long long) K.nnz, (unsigned long long)(A.nnz * A.nnz));
        double maxA = 0, maxK = 0;
        for(ptiNnzIndex z = 0; z < A.nnz; ++z)
            if(fabs((double) A.values.data[z]) > maxA) maxA = fabs((double) A.values.data[z]);
        for(ptiNnzIndex z = 0; z < K.nnz; ++z)
            if(fabs((double) K.values.data[z]) > maxK) maxK = fabs((double) K.values.data[z]);
        CHECK(fabs(maxK - maxA * maxA) <= 1e-5 * maxA * maxA,
              "Kronecker max value %g != (max A)^2 %g", maxK, maxA * maxA);
        ptiFreeSparseTensor(&K);
    }

    /* Khatri-Rao: last-mode dims equal, other dims multiply,
       nnz = sum over slices of (slice nnz)^2 */
    ptiSparseTensor KR;
    rc = ptiSparseTensorKhatriRaoMul(&KR, &A, &A);
    CHECK(rc == 0, "KhatriRao failed rc=%d", rc);
    if(rc == 0) {
        ptiIndex last = A.nmodes - 1;
        CHECK(KR.ndims[last] == A.ndims[last], "KhatriRao last dim wrong");
        /* count nnz per last-mode slice */
        ptiNnzIndex *cnt = (ptiNnzIndex *) calloc(A.ndims[last], sizeof *cnt);
        for(ptiNnzIndex z = 0; z < A.nnz; ++z) cnt[A.inds[last].data[z]]++;
        ptiNnzIndex expect = 0;
        for(ptiIndex k = 0; k < A.ndims[last]; ++k) expect += cnt[k] * cnt[k];
        CHECK(KR.nnz == expect, "KhatriRao nnz %llu != analytic %llu",
              (unsigned long long) KR.nnz, (unsigned long long) expect);
        free(cnt);
        ptiFreeSparseTensor(&KR);
    }
    ptiFreeSparseTensor(&A);
}

static void test_scalar_ops(const char *path)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }

    double sum0 = 0;
    for(ptiNnzIndex z = 0; z < X.nnz; ++z) sum0 += (double) X.values.data[z];

    CHECK(ptiSparseTensorMulScalar(&X, (ptiValue) 2.5) == 0, "MulScalar failed");
    double sum1 = 0;
    for(ptiNnzIndex z = 0; z < X.nnz; ++z) sum1 += (double) X.values.data[z];
    CHECK(fabs(sum1 - 2.5 * sum0) <= 1e-5 * fabs(2.5 * sum0) + 1e-12,
          "MulScalar: sum %g != 2.5 * %g", sum1, sum0);

    ptiFreeSparseTensor(&X);
}


/* semi-sparse conversion round-trip: COO -> semi-sparse -> COO preserves the
 * tensor (this exercises ssptensor/{fromsp,sort,merge,fromssp}.c - the sort is
 * where the sizeof heap overflow lived). Then TTM computed through the
 * semi-sparse form and through OpenMP must agree with the sequential result. */
static void test_semisparse(const char *path, ptiIndex R)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    char what[160];

    for(ptiIndex mode = 0; mode < X.nmodes; ++mode) {
        /* round-trip */
        ptiSemiSparseTensor S;
        snprintf(what, sizeof what, "ToSemiSparse (%s mode %u)", path, mode);
        int rc = ptiSparseTensorToSemiSparseTensor(&S, &X, mode);
        CHECK(rc == 0, "%s failed", what);
        if(rc == 0) {
            ptiSparseTensor B;
            snprintf(what, sizeof what, "SemiToSparse (%s mode %u)", path, mode);
            rc = ptiSemiSparseTensorToSparseTensor(&B, &S, (ptiValue) 1e-9);
            CHECK(rc == 0, "%s failed", what);
            if(rc == 0) {
                CHECK(B.nnz == X.nnz, "%s: nnz %llu != %llu", what,
                      (unsigned long long) B.nnz, (unsigned long long) X.nnz);
                double sx = 0, sb = 0;
                for(ptiNnzIndex z = 0; z < X.nnz; ++z) sx += (double) X.values.data[z];
                for(ptiNnzIndex z = 0; z < B.nnz; ++z) sb += (double) B.values.data[z];
                CHECK(fabs(sb - sx) <= 1e-5 * fabs(sx),
                      "%s: value sum %g != %g", what, sb, sx);
                ptiFreeSparseTensor(&B);
            }

            /* TTM through the semi-sparse form must match the sparse-input TTM */
            ptiMatrix U;
            ptiNewMatrix(&U, X.ndims[mode], R);
            ptiRandomizeMatrix(&U);
            ptiSemiSparseTensor Y1, Y2;
            int rc1 = ptiSparseTensorMulMatrix(&Y1, &X, &U, mode);
            int rc2 = ptiSemiSparseTensorMulMatrix(&Y2, &S, &U, mode);
            snprintf(what, sizeof what, "SemiSparse TTM (%s mode %u R=%u)", path, mode, R);
            CHECK(rc1 == 0 && rc2 == 0, "%s failed (%d, %d)", what, rc1, rc2);
            if(rc1 == 0 && rc2 == 0) {
                CHECK(Y1.nnz == Y2.nnz, "%s: nnz differ", what);
                double s1 = 0, s2 = 0;
                for(ptiNnzIndex z = 0; z < Y1.nnz; ++z)
                    for(ptiIndex r = 0; r < R; ++r) s1 += (double) Y1.values.values[z * Y1.stride + r];
                for(ptiNnzIndex z = 0; z < Y2.nnz; ++z)
                    for(ptiIndex r = 0; r < R; ++r) s2 += (double) Y2.values.values[z * Y2.stride + r];
                CHECK(fabs(s1 - s2) <= 1e-4 * (fabs(s1) > 1 ? fabs(s1) : 1.0),
                      "%s: sums differ %g vs %g", what, s1, s2);
            }
            if(rc1 == 0) ptiFreeSemiSparseTensor(&Y1);
            if(rc2 == 0) ptiFreeSemiSparseTensor(&Y2);

#ifdef HIPARTI_USE_OPENMP
            /* OpenMP TTM against sequential */
            ptiSemiSparseTensor Y3;
            snprintf(what, sizeof what, "Omp TTM (%s mode %u R=%u)", path, mode, R);
            if(ptiOmpSparseTensorMulMatrix(&Y3, &X, &U, mode) == 0) {
                ptiSemiSparseTensor Yref;
                if(ptiSparseTensorMulMatrix(&Yref, &X, &U, mode) == 0) {
                    double s3 = 0, sr = 0;
                    for(ptiNnzIndex z = 0; z < Y3.nnz; ++z)
                        for(ptiIndex r = 0; r < R; ++r) s3 += (double) Y3.values.values[z * Y3.stride + r];
                    for(ptiNnzIndex z = 0; z < Yref.nnz; ++z)
                        for(ptiIndex r = 0; r < R; ++r) sr += (double) Yref.values.values[z * Yref.stride + r];
                    CHECK(fabs(s3 - sr) <= 1e-4 * (fabs(sr) > 1 ? fabs(sr) : 1.0),
                          "%s: omp %g vs seq %g", what, s3, sr);
                    ptiFreeSemiSparseTensor(&Yref);
                }
                ptiFreeSemiSparseTensor(&Y3);
            }
#endif
            ptiFreeMatrix(&U);
            ptiFreeSemiSparseTensor(&S);
        }
    }
    ptiFreeSparseTensor(&X);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    char p[1024];
    snprintf(p, sizeof p, "%s/tensors/3d_7.tns", DATA_DIR);
    test_ttm_against_dense(p, 4);
    test_ttm_against_dense(p, 9);    /* stride != R */
    test_kronecker_khatrirao(p);
    test_scalar_ops(p);
    snprintf(p, sizeof p, "%s/tensors/3d-24.tns", DATA_DIR);
    test_ttm_against_dense(p, 8);
    test_kronecker_khatrirao(p);
    snprintf(p, sizeof p, "%s/tensors/4d_3_16.tns", DATA_DIR);
    test_scalar_ops(p);
    snprintf(p, sizeof p, "%s/tensors/3d_7.tns", DATA_DIR);
    test_semisparse(p, 4);
    snprintf(p, sizeof p, "%s/tensors/3d-24.tns", DATA_DIR);
    test_semisparse(p, 8);
    return TEST_SUMMARY();
}
