/* Boundary sweeps.
 *
 * Most of the defects found in this codebase were rank- or shape-dependent and
 * invisible at the one configuration people habitually run (R=16, 3 modes, a
 * mid-size tensor):
 *   - a GPU kernel wrong only for R = 4, 8, 24  (correct at 16, 32, 64)
 *   - a shared-memory kernel wrong only when stride != R, i.e. R not a multiple of 8
 *   - a code path reached only for nmodes != 3
 * So sweep ranks (especially non-multiples of 8), degenerate shapes, and orders.
 */
#include "test_util.h"

static const char *DATA_DIR;

static void write_tensor(const char *path, ptiIndex nmodes,
                         const ptiIndex *dims, ptiNnzIndex nnz,
                         const ptiIndex *idx /* nnz*nmodes, 1-based */,
                         const double *vals)
{
    FILE *f = fopen(path, "w");
    if(!f) { perror(path); exit(2); }
    fprintf(f, "%u\n", nmodes);
    for(ptiIndex m = 0; m < nmodes; ++m) fprintf(f, "%u%s", dims[m], m + 1 == nmodes ? "\n" : " ");
    for(ptiNnzIndex z = 0; z < nnz; ++z) {
        for(ptiIndex m = 0; m < nmodes; ++m) fprintf(f, "%u ", idx[z * nmodes + m]);
        fprintf(f, "%.9g\n", vals[z]);
    }
    fclose(f);
}

/* run MTTKRP at rank R over every mode and compare against the oracle */
static void mttkrp_at_rank(const ptiSparseTensor *X, ptiIndex R, const char *tag)
{
    for(ptiIndex mode = 0; mode < X->nmodes; ++mode) {
        ptiIndex max_ndims = 0;
        ptiMatrix **U = (ptiMatrix **) malloc((X->nmodes + 1) * sizeof *U);
        for(ptiIndex m = 0; m < X->nmodes; ++m) {
            U[m] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
            ptiNewMatrix(U[m], X->ndims[m], R);
            ptiRandomizeMatrix(U[m]);
            if(X->ndims[m] > max_ndims) max_ndims = X->ndims[m];
        }
        U[X->nmodes] = (ptiMatrix *) malloc(sizeof(ptiMatrix));
        ptiNewMatrix(U[X->nmodes], max_ndims, R);
        ptiConstantMatrix(U[X->nmodes], 0);

        ptiIndex *order = (ptiIndex *) malloc(X->nmodes * sizeof *order);
        order[0] = mode;
        for(ptiIndex i = 1, m = (mode + 1) % X->nmodes; i < X->nmodes; ++i, m = (m + 1) % X->nmodes)
            order[i] = m;

        ptiMatrix ref;
        ptiNewMatrix(&ref, X->ndims[mode], R);
        pti_test_ref_mttkrp(X, U, mode, &ref);

        CHECK(ptiMTTKRP(X, U, order, mode) == 0, "%s: MTTKRP failed (R=%u mode=%u)", tag, R, mode);

        double scale = pti_test_max_abs_matrix(&ref), worst = 0;
        for(ptiIndex i = 0; i < X->ndims[mode]; ++i)
            for(ptiIndex r = 0; r < R; ++r) {
                double d = fabs((double) U[X->nmodes]->values[i * U[X->nmodes]->stride + r]
                              - (double) ref.values[i * ref.stride + r]);
                if(d > worst) worst = d;
            }
        CHECK(worst <= 2e-5 * (scale > 0 ? scale : 1.0),
              "%s: R=%u mode=%u max abs diff %g (scale %g)", tag, R, mode, worst, scale);

        /* stride is rounded up to a multiple of 8; make sure nothing wrote past ncols */
        CHECK(U[X->nmodes]->stride >= R, "%s: stride %u < R %u", tag, U[X->nmodes]->stride, R);

        ptiFreeMatrix(&ref); free(order);
        for(ptiIndex m = 0; m <= X->nmodes; ++m) { ptiFreeMatrix(U[m]); free(U[m]); }
        free(U);
    }
}

/* the rank sweep deliberately includes values where stride != R */
static void test_rank_sweep(void)
{
    char p[1024];
    snprintf(p, sizeof p, "%s/tensors/3d_7.tns", DATA_DIR);
    ptiSparseTensor X;
    if(pti_test_load(&X, p) != 0) { ++pti_test_failures; return; }
    const ptiIndex ranks[] = { 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 24, 31, 32, 64 };
    for(size_t k = 0; k < sizeof(ranks)/sizeof(*ranks); ++k)
        mttkrp_at_rank(&X, ranks[k], "rank-sweep");
    ptiFreeSparseTensor(&X);
}

/* a single nonzero: smallest possible non-empty tensor */
static void test_single_nonzero(void)
{
    const char *p = "test_edge_single.tns";
    ptiIndex dims[3] = { 4, 4, 3 };
    ptiIndex idx[3]  = { 2, 3, 1 };
    double   val[1]  = { 2.5 };
    write_tensor(p, 3, dims, 1, idx, val);

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensor(&X, 1, (char *) p) == 0, "single-nnz load failed");
    CHECK(X.nnz == 1, "single-nnz: nnz = %llu", (unsigned long long) X.nnz);
    mttkrp_at_rank(&X, 4, "single-nnz");
    mttkrp_at_rank(&X, 17, "single-nnz");
    ptiFreeSparseTensor(&X);
    remove(p);
}

/* a mode of extent 1 - trivially degenerate, but it changes block/grid maths */
static void test_singleton_dimension(void)
{
    const char *p = "test_edge_dim1.tns";
    ptiIndex dims[3] = { 4, 1, 3 };
    ptiIndex idx[4 * 3] = { 1,1,1,  2,1,2,  3,1,3,  4,1,1 };
    double   val[4] = { 1.0, -2.0, 3.5, 4.25 };
    write_tensor(p, 3, dims, 4, idx, val);

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensor(&X, 1, (char *) p) == 0, "dim-1 load failed");
    CHECK(X.ndims[1] == 1, "dim-1: ndims[1] = %u", X.ndims[1]);
    mttkrp_at_rank(&X, 8, "dim-1");
    ptiFreeSparseTensor(&X);
    remove(p);
}

/* orders other than 3, since MTTKRP special-cases nmodes==3 */
static void test_order_sweep(void)
{
    for(ptiIndex nm = 2; nm <= 5; ++nm) {
        char p[128];
        snprintf(p, sizeof p, "test_edge_order%u.tns", nm);

        ptiIndex dims[5];
        for(ptiIndex m = 0; m < nm; ++m) dims[m] = 3 + m;

        ptiNnzIndex const nnz = 12;
        ptiIndex *idx = (ptiIndex *) malloc(nnz * nm * sizeof *idx);
        double   *val = (double *)   malloc(nnz * sizeof *val);
        uint64_t rng = 7777ULL;
        for(ptiNnzIndex z = 0; z < nnz; ++z) {
            for(ptiIndex m = 0; m < nm; ++m)
                idx[z * nm + m] = 1 + (ptiIndex)((z * (m + 2) + m) % dims[m]);
            val[z] = (double) pti_test_rand(&rng) + 0.5;
        }
        write_tensor(p, nm, dims, nnz, idx, val);

        ptiSparseTensor X;
        char tag[64]; snprintf(tag, sizeof tag, "order-%u", nm);
        if(ptiLoadSparseTensor(&X, 1, (char *) p) == 0) {
            CHECK(X.nmodes == nm, "%s: nmodes = %u", tag, X.nmodes);
            mttkrp_at_rank(&X, 4, tag);
            mttkrp_at_rank(&X, 16, tag);
            ptiFreeSparseTensor(&X);
        } else {
            CHECK(0, "%s: load failed", tag);
        }
        free(idx); free(val);
        remove(p);
    }
}

/* repeated coordinates: the reference sums them, so the library must too */
static void test_duplicate_indices(void)
{
    const char *p = "test_edge_dup.tns";
    ptiIndex dims[3] = { 3, 3, 2 };
    ptiIndex idx[4 * 3] = { 1,1,1,  1,1,1,  2,2,2,  1,1,1 };
    double   val[4] = { 1.0, 2.0, 5.0, 3.0 };
    write_tensor(p, 3, dims, 4, idx, val);

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensor(&X, 1, (char *) p) == 0, "dup load failed");
    /* the oracle accumulates duplicates, so agreement is the whole check here */
    mttkrp_at_rank(&X, 4, "duplicate-indices");
    ptiFreeSparseTensor(&X);
    remove(p);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    test_rank_sweep();
    test_single_nonzero();
    test_singleton_dimension();
    test_order_sweep();
    test_duplicate_indices();
    return TEST_SUMMARY();
}
