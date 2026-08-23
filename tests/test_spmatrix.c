/* Sparse matrix operations: SpMV and SpMM in COO, CSR, and HiCOO form,
 * sequential and OpenMP, all checked against a dense reference computed from
 * the loaded COO data - plus the structural properties of format conversion
 * and relabelling.  Until now none of these paths had any numeric check.
 */
#include "test_util.h"

static const char *DATA_DIR;

/* dense y = A*x from raw COO triples */
static void ref_spmv(const ptiSparseMatrix *A, const ptiValueVector *x, double *y)
{
    memset(y, 0, A->nrows * sizeof *y);
    for(ptiNnzIndex z = 0; z < A->nnz; ++z)
        y[A->rowind.data[z]] += (double) A->values.data[z] * (double) x->data[A->colind.data[z]];
}

/* dense C = A*B from raw COO triples */
static void ref_spmm(const ptiSparseMatrix *A, const ptiMatrix *B, double *C, ptiIndex R)
{
    memset(C, 0, (size_t) A->nrows * R * sizeof *C);
    for(ptiNnzIndex z = 0; z < A->nnz; ++z) {
        ptiIndex const i = A->rowind.data[z], j = A->colind.data[z];
        for(ptiIndex r = 0; r < R; ++r)
            C[(size_t) i * R + r] += (double) A->values.data[z] * (double) B->values[j * B->stride + r];
    }
}

static void fill_vector(ptiValueVector *x, uint64_t seed)
{
    uint64_t rng = seed;
    for(uint64_t i = 0; i < x->len; ++i) x->data[i] = pti_test_rand(&rng);
}

static void check_vec(const ptiValueVector *y, const double *ref, ptiIndex n, const char *what)
{
    double scale = 0, worst = 0;
    for(ptiIndex i = 0; i < n; ++i) {
        if(fabs(ref[i]) > scale) scale = fabs(ref[i]);
        double d = fabs((double) y->data[i] - ref[i]);
        if(d > worst) worst = d;
    }
    CHECK(!(scale > 0 && worst == scale), "%s: result looks untouched (all zeros?)", what);
    CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0), "%s: max abs diff %g (scale %g)", what, worst, scale);
}

static void check_mat(const ptiMatrix *C, const double *ref, ptiIndex nrows, ptiIndex R, const char *what)
{
    double scale = 0, worst = 0;
    for(ptiIndex i = 0; i < nrows; ++i)
        for(ptiIndex r = 0; r < R; ++r) {
            double e = ref[(size_t) i * R + r];
            if(fabs(e) > scale) scale = fabs(e);
            double d = fabs((double) C->values[i * C->stride + r] - e);
            if(d > worst) worst = d;
        }
    CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0), "%s: max abs diff %g (scale %g)", what, worst, scale);
}

static int load_mtx(ptiSparseMatrix *A, const char *name)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/matrices/%s", DATA_DIR, name);
    FILE *f = fopen(p, "r");
    if(!f) { fprintf(stderr, "  cannot open %s\n", p); return -1; }
    int rc = ptiLoadSparseMatrix(A, 1, f);
    fclose(f);
    if(rc != 0) fprintf(stderr, "  cannot parse %s\n", p);
    return rc;
}

static void test_spmv_all(const char *name)
{
    ptiSparseMatrix A;
    if(load_mtx(&A, name) != 0) { ++pti_test_failures; return; }
    char what[160];

    ptiValueVector x, y;
    ptiNewValueVector(&x, A.ncols, A.ncols);
    fill_vector(&x, 42);
    double *ref = (double *) malloc(A.nrows * sizeof *ref);
    ref_spmv(&A, &x, ref);

    /* COO sequential */
    ptiNewValueVector(&y, A.nrows, A.nrows);
    ptiConstantValueVector(&y, 0);
    snprintf(what, sizeof what, "SpMV COO seq (%s)", name);
    CHECK(ptiSparseMatrixMulVector(&y, &A, &x) == 0, "%s failed", what);
    check_vec(&y, ref, A.nrows, what);

#ifdef HIPARTI_USE_OPENMP
    ptiConstantValueVector(&y, 0);
    snprintf(what, sizeof what, "SpMV COO omp (%s)", name);
    CHECK(ptiOmpSparseMatrixMulVector(&y, &A, &x) == 0, "%s failed", what);
    check_vec(&y, ref, A.nrows, what);
#endif

    /* CSR */
    ptiSparseMatrixCSR csr;
    snprintf(what, sizeof what, "COO->CSR (%s)", name);
    CHECK(ptiSparseMatrixToCSR(&csr, &A) == 0, "%s failed", what);
    CHECK(csr.nnz == A.nnz && csr.nrows == A.nrows && csr.ncols == A.ncols,
          "%s changed shape/nnz", what);
    ptiConstantValueVector(&y, 0);
    snprintf(what, sizeof what, "SpMV CSR seq (%s)", name);
    CHECK(ptiSparseMatrixMulVectorCSR(&y, &csr, &x) == 0, "%s failed", what);
    check_vec(&y, ref, A.nrows, what);
#ifdef HIPARTI_USE_OPENMP
    ptiConstantValueVector(&y, 0);
    snprintf(what, sizeof what, "SpMV CSR omp (%s)", name);
    CHECK(ptiOmpSparseMatrixMulVectorCSR(&y, &csr, &x) == 0, "%s failed", what);
    check_vec(&y, ref, A.nrows, what);
#endif
    ptiFreeSparseMatrixCSR(&csr);

    /* HiCOO */
    ptiSparseMatrixHiCOO H;
    ptiNnzIndex max_nnzb = 0;
    snprintf(what, sizeof what, "COO->HiCOO mat (%s)", name);
    /* note: conversion sorts A in place; the COO reference above is order-independent */
    if(ptiSparseMatrixToHiCOO(&H, &max_nnzb, &A, 7, 10) == 0) {
        CHECK(H.nnz == A.nnz, "%s changed nnz", what);
        ptiConstantValueVector(&y, 0);
        snprintf(what, sizeof what, "SpMV HiCOO seq (%s)", name);
        CHECK(ptiSparseMatrixMulVectorHiCOO(&y, &H, &x) == 0, "%s failed", what);
        check_vec(&y, ref, A.nrows, what);
#ifdef HIPARTI_USE_OPENMP
        ptiConstantValueVector(&y, 0);
        snprintf(what, sizeof what, "SpMV HiCOO omp (%s)", name);
        if(ptiOmpSparseMatrixMulVectorHiCOO(&y, &H, &x) == 0)
            check_vec(&y, ref, A.nrows, what);
#endif
        ptiFreeSparseMatrixHiCOO(&H);
    } else {
        CHECK(0, "%s failed", what);
    }

    free(ref);
    ptiFreeValueVector(&x);
    ptiFreeValueVector(&y);
    ptiFreeSparseMatrix(&A);
}

static void test_spmm_all(const char *name, ptiIndex R)
{
    ptiSparseMatrix A;
    if(load_mtx(&A, name) != 0) { ++pti_test_failures; return; }
    char what[160];

    ptiMatrix B, C;
    ptiNewMatrix(&B, A.ncols, R);
    ptiRandomizeMatrix(&B);
    double *ref = (double *) malloc((size_t) A.nrows * R * sizeof *ref);
    ref_spmm(&A, &B, ref, R);

    ptiNewMatrix(&C, A.nrows, R);

    ptiConstantMatrix(&C, 0);
    snprintf(what, sizeof what, "SpMM COO seq (%s R=%u)", name, R);
    CHECK(ptiSparseMatrixMulMatrix(&C, &A, &B) == 0, "%s failed", what);
    check_mat(&C, ref, A.nrows, R, what);

#ifdef HIPARTI_USE_OPENMP
    ptiConstantMatrix(&C, 0);
    snprintf(what, sizeof what, "SpMM COO omp (%s R=%u)", name, R);
    CHECK(ptiOmpSparseMatrixMulMatrix(&C, &A, &B) == 0, "%s failed", what);
    check_mat(&C, ref, A.nrows, R, what);
#endif

    ptiSparseMatrixCSR csr;
    if(ptiSparseMatrixToCSR(&csr, &A) == 0) {
        ptiConstantMatrix(&C, 0);
        snprintf(what, sizeof what, "SpMM CSR seq (%s R=%u)", name, R);
        CHECK(ptiSparseMatrixMulMatrixCSR(&C, &csr, &B) == 0, "%s failed", what);
        check_mat(&C, ref, A.nrows, R, what);
#ifdef HIPARTI_USE_OPENMP
        ptiConstantMatrix(&C, 0);
        snprintf(what, sizeof what, "SpMM CSR omp (%s R=%u)", name, R);
        CHECK(ptiOmpSparseMatrixMulMatrixCSR(&C, &csr, &B) == 0, "%s failed", what);
        check_mat(&C, ref, A.nrows, R, what);
#endif
        ptiFreeSparseMatrixCSR(&csr);
    }

    ptiSparseMatrixHiCOO H;
    ptiNnzIndex max_nnzb = 0;
    if(ptiSparseMatrixToHiCOO(&H, &max_nnzb, &A, 7, 10) == 0) {
        ptiConstantMatrix(&C, 0);
        snprintf(what, sizeof what, "SpMM HiCOO seq (%s R=%u)", name, R);
        CHECK(ptiSparseMatrixMulMatrixHiCOO(&C, &H, &B) == 0, "%s failed", what);
        check_mat(&C, ref, A.nrows, R, what);
        ptiFreeSparseMatrixHiCOO(&H);
    }

    free(ref);
    ptiFreeMatrix(&B); ptiFreeMatrix(&C);
    ptiFreeSparseMatrix(&A);
}

/* relabelling must be a bijection per mode, and SpMV on the relabelled matrix
 * with correspondingly permuted x must give the permuted y */
static void test_relabel(const char *name)
{
    ptiSparseMatrix A;
    if(load_mtx(&A, name) != 0) { ++pti_test_failures; return; }

    ptiValueVector x, y;
    ptiNewValueVector(&x, A.ncols, A.ncols);
    fill_vector(&x, 7);
    double *y0 = (double *) malloc(A.nrows * sizeof *y0);
    ref_spmv(&A, &x, y0);

    ptiIndex *map[2];
    ptiIndex dims[2] = { A.nrows, A.ncols };
    for(int m = 0; m < 2; ++m) {
        map[m] = (ptiIndex *) malloc(dims[m] * sizeof(ptiIndex));
        for(ptiIndex i = 0; i < dims[m]; ++i) map[m][i] = i;
    }
    ptiIndexRelabel(&A, map, 1 /* Lexi-order */, 3, 1);

    for(int m = 0; m < 2; ++m) {
        char *seen = (char *) calloc(dims[m], 1);
        int bij = 1;
        for(ptiIndex i = 0; i < dims[m]; ++i) {
            if(map[m][i] >= dims[m] || seen[map[m][i]]) { bij = 0; break; }
            seen[map[m][i]] = 1;
        }
        CHECK(bij, "relabel mode %d is not a bijection (%s)", m, name);
        free(seen);
    }

    /* permute the matrix and the vector, recompute, un-permute, compare */
    ptiSparseMatrixShuffleIndices(&A, map);
    ptiValueVector xp;
    ptiNewValueVector(&xp, A.ncols, A.ncols);
    for(ptiIndex j = 0; j < A.ncols; ++j) xp.data[map[1][j]] = x.data[j];
    ptiNewValueVector(&y, A.nrows, A.nrows);
    ptiConstantValueVector(&y, 0);
    CHECK(ptiSparseMatrixMulVector(&y, &A, &xp) == 0, "SpMV after relabel failed (%s)", name);
    double scale = 0, worst = 0;
    for(ptiIndex i = 0; i < A.nrows; ++i) {
        if(fabs(y0[i]) > scale) scale = fabs(y0[i]);
        double d = fabs((double) y.data[map[0][i]] - y0[i]);
        if(d > worst) worst = d;
    }
    CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0),
          "relabelled SpMV disagrees after un-permutation (%s): diff %g scale %g", name, worst, scale);

    free(y0); free(map[0]); free(map[1]);
    ptiFreeValueVector(&x); ptiFreeValueVector(&xp); ptiFreeValueVector(&y);
    ptiFreeSparseMatrix(&A);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    const char *mats[] = { "tiny.mtx", "1138_bus.mtx" };
    for(size_t i = 0; i < sizeof(mats)/sizeof(*mats); ++i) {
        test_spmv_all(mats[i]);
        test_spmm_all(mats[i], 8);
        test_spmm_all(mats[i], 9);   /* stride != R */
        test_relabel(mats[i]);
    }
    return TEST_SUMMARY();
}
