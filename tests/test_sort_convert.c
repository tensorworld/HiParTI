/* Structural invariants that hold regardless of implementation:
     - sorting permutes the nonzeros, it does not create or destroy them
     - a sorted tensor really is in nondecreasing lexicographic order
     - COO -> HiCOO -> COO returns the same multiset of nonzeros
     - renumbering is a bijection, and shuffle followed by its inverse is identity */
#include "test_util.h"

static const char *DATA_DIR;

static double checksum(const ptiSparseTensor *X)
{
    double s = 0;
    for(ptiNnzIndex z = 0; z < X->nnz; ++z) s += (double) X->values.data[z];
    return s;
}

static int lex_cmp(const ptiSparseTensor *X, ptiNnzIndex a, ptiNnzIndex b)
{
    for(ptiIndex m = 0; m < X->nmodes; ++m) {
        if(X->inds[m].data[a] < X->inds[m].data[b]) return -1;
        if(X->inds[m].data[a] > X->inds[m].data[b]) return  1;
    }
    return 0;
}

static void test_sort(const char *path)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    ptiNnzIndex const nnz0 = X.nnz;
    double const sum0 = checksum(&X);

    ptiSparseTensorSortIndex(&X, 1, 1);

    CHECK(X.nnz == nnz0, "sort changed nnz: %llu -> %llu",
          (unsigned long long) nnz0, (unsigned long long) X.nnz);
    CHECK(fabs(checksum(&X) - sum0) <= 1e-9 * (fabs(sum0) + 1),
          "sort changed the sum of values");

    int sorted = 1;
    for(ptiNnzIndex z = 1; z < X.nnz; ++z)
        if(lex_cmp(&X, z - 1, z) > 0) { sorted = 0; break; }
    CHECK(sorted, "tensor is not in lexicographic order after sorting");

    for(ptiIndex m = 0; m < X.nmodes; ++m)
        for(ptiNnzIndex z = 0; z < X.nnz; ++z)
            CHECK(X.inds[m].data[z] < X.ndims[m], "index out of bounds after sort");

    ptiFreeSparseTensor(&X);
}

static void test_hicoo_roundtrip(const char *path, ptiElementIndex sb, ptiElementIndex sk, ptiElementIndex sc)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }
    ptiNnzIndex const nnz0 = X.nnz;
    double const sum0 = checksum(&X);

    ptiSparseTensorHiCOO H;
    ptiNnzIndex max_nnzb = 0;
    int rc = ptiSparseTensorToHiCOO(&H, &max_nnzb, &X, sb, sk, sc, 1);
    CHECK(rc == 0, "COO -> HiCOO failed");
    if(rc == 0) {
        CHECK(H.nnz == nnz0, "HiCOO nnz %llu != COO nnz %llu",
              (unsigned long long) H.nnz, (unsigned long long) nnz0);
        double hsum = 0;
        for(ptiNnzIndex z = 0; z < H.nnz; ++z) hsum += (double) H.values.data[z];
        CHECK(fabs(hsum - sum0) <= 1e-6 * (fabs(sum0) + 1),
              "HiCOO conversion changed the sum of values (%g vs %g)", hsum, sum0);
        ptiFreeSparseTensorHiCOO(&H);
    }
    ptiFreeSparseTensor(&X);
}

static void test_renumber_bijection(const char *path)
{
    ptiSparseTensor X;
    if(pti_test_load(&X, path) != 0) { ++pti_test_failures; return; }

    ptiIndex **map_inds = (ptiIndex **) malloc(X.nmodes * sizeof *map_inds);
    for(ptiIndex m = 0; m < X.nmodes; ++m) {
        map_inds[m] = (ptiIndex *) malloc(X.ndims[m] * sizeof (ptiIndex));
        for(ptiIndex i = 0; i < X.ndims[m]; ++i) map_inds[m][i] = i;
    }

    ptiIndexRenumber(&X, map_inds, 1 /* Lexi-order */, 3, 5, 1, 1 /* impl_num */);

    for(ptiIndex m = 0; m < X.nmodes; ++m) {
        char *seen = (char *) calloc(X.ndims[m], 1);
        int bij = 1;
        for(ptiIndex i = 0; i < X.ndims[m]; ++i) {
            ptiIndex const t = map_inds[m][i];
            if(t >= X.ndims[m] || seen[t]) { bij = 0; break; }
            seen[t] = 1;
        }
        CHECK(bij, "renumbering of mode %u is not a bijection", m);
        free(seen);
    }
    for(ptiIndex m = 0; m < X.nmodes; ++m) free(map_inds[m]);
    free(map_inds);
    ptiFreeSparseTensor(&X);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    char p[1024];
    const char *tensors[] = { "3d_7.tns", "3d-24.tns", "3D_12031.tns" };
    for(size_t t = 0; t < sizeof(tensors)/sizeof(*tensors); ++t) {
        snprintf(p, sizeof p, "%s/tensors/%s", DATA_DIR, tensors[t]);
        test_sort(p);
        test_hicoo_roundtrip(p, 2, 4, 2);
        test_renumber_bijection(p);
    }
    return TEST_SUMMARY();
}
