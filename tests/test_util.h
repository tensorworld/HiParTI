/* Shared helpers for the HiParTI test suite. */
#ifndef PTI_TEST_UTIL_H
#define PTI_TEST_UTIL_H

#include <HiParTI.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int pti_test_failures = 0;
static int pti_test_checks   = 0;

#define CHECK(cond, ...) do {                                              \
    ++pti_test_checks;                                                     \
    if(!(cond)) {                                                          \
        ++pti_test_failures;                                               \
        fprintf(stderr, "  FAIL %s:%d: ", __FILE__, __LINE__);             \
        fprintf(stderr, __VA_ARGS__);                                      \
        fprintf(stderr, "\n");                                             \
    }                                                                      \
} while(0)

#define TEST_SUMMARY() (                                                   \
    fprintf(stderr, "%s: %d/%d checks passed\n",                           \
            pti_test_failures ? "FAILED" : "ok",                           \
            pti_test_checks - pti_test_failures, pti_test_checks),         \
    pti_test_failures ? 1 : 0)

/* Same splitmix64 as ptiRandomizeMatrix, so tests can predict factor values. */
static inline ptiValue pti_test_rand(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return (ptiValue)((double)(z >> 11) / 9007199254740992.0);
}

/* Relative comparison; float32 accumulation and GPU atomic reordering both
   perturb the low bits, so exact equality is not a meaningful criterion. */
static inline int pti_test_close(double got, double exp_, double scale)
{
    double tol = 2e-5 * (scale > 0 ? scale : 1.0);
    return fabs(got - exp_) <= tol;
}

static inline double pti_test_max_abs_matrix(const ptiMatrix *m)
{
    double mx = 0;
    for(ptiIndex i = 0; i < m->nrows; ++i)
        for(ptiIndex j = 0; j < m->ncols; ++j) {
            double v = fabs((double) m->values[i * m->stride + j]);
            if(v > mx) mx = v;
        }
    return mx;
}

/* Straightforward MTTKRP, written for clarity rather than speed: it is the
   oracle the optimised kernels are checked against. */
static inline void pti_test_ref_mttkrp(
    const ptiSparseTensor *X, ptiMatrix ** U, ptiIndex mode, ptiMatrix *out)
{
    ptiIndex const R = out->ncols;
    for(ptiIndex i = 0; i < out->nrows; ++i)
        for(ptiIndex r = 0; r < R; ++r)
            out->values[i * out->stride + r] = 0;

    for(ptiNnzIndex z = 0; z < X->nnz; ++z) {
        ptiIndex const row = X->inds[mode].data[z];
        for(ptiIndex r = 0; r < R; ++r) {
            double prod = (double) X->values.data[z];
            for(ptiIndex m = 0; m < X->nmodes; ++m) {
                if(m == mode) continue;
                ptiIndex const im = X->inds[m].data[z];
                prod *= (double) U[m]->values[im * U[m]->stride + r];
            }
            out->values[row * out->stride + r] += (ptiValue) prod;
        }
    }
}

static inline int pti_test_load(ptiSparseTensor *X, const char *path)
{
    if(ptiLoadSparseTensor(X, 1, (char *) path) != 0) {
        fprintf(stderr, "  cannot load %s\n", path);
        return -1;
    }
    return 0;
}

#endif /* PTI_TEST_UTIL_H */
