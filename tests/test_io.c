/* I/O: native .tns round-trip, and the TensorSuite reader (both index bases,
   pattern-only files, both fill modes, and rejection of block-sparse input). */
#include "test_util.h"

static const char *DATA_DIR;

static void write_file(const char *path, const char *body)
{
    FILE *f = fopen(path, "w");
    if(!f) { perror(path); exit(2); }
    fputs(body, f);
    fclose(f);
}

static void test_native_roundtrip(void)
{
    char in[1024];
    snprintf(in, sizeof(in), "%s/tensors/3d_7.tns", DATA_DIR);

    ptiSparseTensor X;
    if(pti_test_load(&X, in) != 0) { ++pti_test_failures; return; }

    const char *tmp = "test_io_native.tns";
    FILE *f = fopen(tmp, "w");
    CHECK(f != NULL, "cannot open %s for writing", tmp);
    if(!f) return;
    CHECK(ptiDumpSparseTensor(&X, 1, f) == 0, "dump failed");
    fclose(f);

    ptiSparseTensor Y;
    CHECK(ptiLoadSparseTensor(&Y, 1, (char *) tmp) == 0, "reload failed");
    CHECK(X.nmodes == Y.nmodes, "nmodes %u != %u", X.nmodes, Y.nmodes);
    CHECK(X.nnz == Y.nnz, "nnz differs after round-trip");
    for(ptiIndex m = 0; m < X.nmodes && m < Y.nmodes; ++m)
        CHECK(X.ndims[m] == Y.ndims[m], "dim %u differs", m);
    if(X.nnz == Y.nnz) {
        int same = 1;
        for(ptiNnzIndex z = 0; z < X.nnz; ++z) {
            for(ptiIndex m = 0; m < X.nmodes; ++m)
                if(X.inds[m].data[z] != Y.inds[m].data[z]) same = 0;
            if(X.values.data[z] != Y.values.data[z]) same = 0;
        }
        CHECK(same, "round-trip changed the tensor contents");
    }
    ptiFreeSparseTensor(&X);
    ptiFreeSparseTensor(&Y);
    remove(tmp);
}

/* 1-based, no value column -> values synthesised */
static void test_tensorsuite_base1_pattern(void)
{
    const char *p = "test_io_ts_b1.tns";
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% version: 0.1\n"
        "% name: unit-b1\n"
        "3 4 4 3 5\n"
        "1 1 1\n"
        "1 2 1\n"
        "2 1 2\n"
        "3 3 1\n"
        "4 4 3\n");

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensorTensorSuite(&X, 1, p, PTI_TS_FILL_ONES) == 0, "load failed");
    CHECK(X.nmodes == 3, "nmodes %u != 3", X.nmodes);
    CHECK(X.nnz == 5, "nnz %llu != 5", (unsigned long long) X.nnz);
    CHECK(X.ndims[0] == 4 && X.ndims[1] == 4 && X.ndims[2] == 3, "dims wrong");
    /* stored 1-based, so internally 0-based: first entry becomes (0,0,0) */
    CHECK(X.inds[0].data[0] == 0 && X.inds[1].data[0] == 0 && X.inds[2].data[0] == 0,
          "index base not removed: got (%u,%u,%u)",
          X.inds[0].data[0], X.inds[1].data[0], X.inds[2].data[0]);
    /* last entry 4 4 3 -> (3,3,2), i.e. exactly on the dimension bounds */
    CHECK(X.inds[0].data[4] == 3 && X.inds[1].data[4] == 3 && X.inds[2].data[4] == 2,
          "last index wrong");
    int all_ones = 1;
    for(ptiNnzIndex z = 0; z < X.nnz; ++z) if(X.values.data[z] != (ptiValue) 1) all_ones = 0;
    CHECK(all_ones, "FILL_ONES did not produce all 1.0");
    ptiFreeSparseTensor(&X);
    remove(p);
}

/* 0-based indices must be detected and left alone */
static void test_tensorsuite_base0(void)
{
    const char *p = "test_io_ts_b0.tns";
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% name: unit-b0\n"
        "3 4 4 3 3\n"
        "0 0 0\n"
        "1 2 1\n"
        "3 3 2\n");

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensorTensorSuite(&X, 1, p, PTI_TS_FILL_ONES) == 0, "load failed");
    CHECK(X.nnz == 3, "nnz wrong");
    /* a 0 index proves base 0, so indices must pass through unchanged */
    CHECK(X.inds[0].data[0] == 0 && X.inds[0].data[2] == 3,
          "0-based indices were shifted: got %u and %u",
          X.inds[0].data[0], X.inds[0].data[2]);
    for(ptiIndex m = 0; m < X.nmodes; ++m)
        for(ptiNnzIndex z = 0; z < X.nnz; ++z)
            CHECK(X.inds[m].data[z] < X.ndims[m], "index out of range after rebase");
    ptiFreeSparseTensor(&X);
    remove(p);
}

/* an explicit value column must be honoured, not overwritten by the fill */
static void test_tensorsuite_with_values(void)
{
    const char *p = "test_io_ts_val.tns";
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% name: unit-val\n"
        "3 4 4 3 3\n"
        "1 1 1 1.5\n"
        "1 2 1 -2.25\n"
        "2 1 2 4\n");

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensorTensorSuite(&X, 1, p, PTI_TS_FILL_ONES) == 0, "load failed");
    CHECK(X.nnz == 3, "nnz wrong");
    CHECK(X.values.data[0] == (ptiValue) 1.5,   "value 0 = %g", (double) X.values.data[0]);
    CHECK(X.values.data[1] == (ptiValue) -2.25, "value 1 = %g", (double) X.values.data[1]);
    CHECK(X.values.data[2] == (ptiValue) 4.0,   "value 2 = %g", (double) X.values.data[2]);
    ptiFreeSparseTensor(&X);
    remove(p);
}

/* FILL_RANDOM must be varied but reproducible run to run */
static void test_tensorsuite_fill_random(void)
{
    const char *p = "test_io_ts_rnd.tns";
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% name: unit-rnd\n"
        "3 4 4 3 4\n"
        "1 1 1\n1 2 1\n2 1 2\n3 3 1\n");

    ptiSparseTensor A, B;
    CHECK(ptiLoadSparseTensorTensorSuite(&A, 1, p, PTI_TS_FILL_RANDOM) == 0, "load A failed");
    CHECK(ptiLoadSparseTensorTensorSuite(&B, 1, p, PTI_TS_FILL_RANDOM) == 0, "load B failed");

    int identical = 1, all_same_value = 1;
    for(ptiNnzIndex z = 0; z < A.nnz && z < B.nnz; ++z) {
        if(A.values.data[z] != B.values.data[z]) identical = 0;
        if(A.values.data[z] != A.values.data[0]) all_same_value = 0;
    }
    CHECK(identical, "FILL_RANDOM is not reproducible across loads");
    CHECK(!all_same_value, "FILL_RANDOM produced a constant vector");
    for(ptiNnzIndex z = 0; z < A.nnz; ++z)
        CHECK(A.values.data[z] >= 0 && A.values.data[z] < 1, "random value out of [0,1)");
    ptiFreeSparseTensor(&A);
    ptiFreeSparseTensor(&B);
    remove(p);
}

/* block-sparse must be refused rather than silently misread */
static void test_tensorsuite_block_rejected(void)
{
    const char *p = "test_io_ts_blk.tns";
    const char *j = "test_io_ts_blk_metadata.json";
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% name: unit-blk\n"
        "4 20 20 4 4 119\n"
        "19 19 4 4 101\n"
        "0 0 0 0\n");
    write_file(j, "{\n  \"sparsity_type\": \"block\",\n  \"index_base\": 0\n}\n");

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensorTensorSuite(&X, 1, p, PTI_TS_FILL_ONES) != 0,
          "block-sparse input was accepted; it should be rejected");
    remove(p);
    remove(j);
}

/* the sidecar metadata must win over inference */
static void test_tensorsuite_metadata_base(void)
{
    const char *p = "test_io_ts_meta.tns";
    const char *j = "test_io_ts_meta_metadata.json";
    /* no 0 appears, so inference would guess base 1 - metadata says otherwise */
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% name: unit-meta\n"
        "3 4 4 3 2\n"
        "1 1 1\n2 2 2\n");
    write_file(j, "{\n  \"sparsity_type\": \"element\",\n  \"index_base\": 0\n}\n");

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensorTensorSuite(&X, 1, p, PTI_TS_FILL_ONES) == 0, "load failed");
    CHECK(X.inds[0].data[0] == 1, "metadata index_base=0 ignored (got %u, expected 1)",
          X.inds[0].data[0]);
    ptiFreeSparseTensor(&X);
    remove(p);
    remove(j);
}

/* ptiLoadSparseTensor must recognise the banner on its own */
static void test_autodetect(void)
{
    const char *p = "test_io_auto.tns";
    write_file(p,
        "%%TensorSuite-TNS\n"
        "% name: unit-auto\n"
        "3 4 4 3 2\n"
        "1 1 1\n2 2 2\n");

    ptiSparseTensor X;
    CHECK(ptiLoadSparseTensor(&X, 1, (char *) p) == 0,
          "ptiLoadSparseTensor did not auto-detect the TensorSuite banner");
    CHECK(X.nnz == 2, "nnz wrong after auto-detect");
    ptiFreeSparseTensor(&X);
    remove(p);
}


/* the error-reporting API: provoke a shape error and read it back */
static void test_error_api(void)
{
    ptiSparseTensor A, B, Z;
    ptiIndex da[2] = { 3, 3 }, db[2] = { 4, 4 };
    ptiNewSparseTensor(&A, 2, da);
    ptiNewSparseTensor(&B, 2, db);
    ptiClearLastError();
    CHECK(ptiSparseTensorDotMul(&Z, &A, &B) != 0, "shape mismatch was accepted");
    const char *module = NULL, *file = NULL, *reason = NULL;
    unsigned line = 0;
    /* pti_CheckError records the error only in HIPARTI_DEBUG builds, so the
       code may legitimately be 0 here in a release build; the API calls
       themselves must still work and Clear must always leave it at 0. */
    (void) ptiGetLastError(&module, &file, &line, &reason);
    ptiClearLastError();
    int code = ptiGetLastError(&module, &file, &line, &reason);
    CHECK(code == 0, "ptiClearLastError did not clear the error");
    ptiFreeSparseTensor(&A);
    ptiFreeSparseTensor(&B);
}

int main(int argc, char **argv)
{
    DATA_DIR = (argc > 1) ? argv[1] : "data";
    test_native_roundtrip();
    test_tensorsuite_base1_pattern();
    test_tensorsuite_base0();
    test_tensorsuite_with_values();
    test_tensorsuite_fill_random();
    test_tensorsuite_block_rejected();
    test_tensorsuite_metadata_base();
    test_autodetect();
    test_error_api();
    return TEST_SUMMARY();
}
