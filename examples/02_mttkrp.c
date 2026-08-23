/* Example 2: MTTKRP - the workhorse of CP decomposition.
 *
 *   ./02_mttkrp ../data/tensors/3d_7.tns 0 8       (tensor, mode, rank)
 *
 * Covers: factor-matrix setup, mode ordering, ptiMTTKRP and the OpenMP
 * variant, and how results come back in mats[nmodes].
 */
#include <HiParTI.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if(argc < 4) {
        fprintf(stderr, "usage: %s tensor.tns mode rank\n", argv[0]);
        return 1;
    }
    ptiIndex const mode = (ptiIndex) atoi(argv[2]);
    ptiIndex const R = (ptiIndex) atoi(argv[3]);

    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, argv[1]) != 0) return 1;
    ptiIndex const nmodes = X.nmodes;
    if(mode >= nmodes) { fprintf(stderr, "mode out of range\n"); return 1; }

    /* One factor matrix per mode, plus one scratch/output matrix at the end
       sized by the largest dimension.  ptiRandomizeMatrix is deterministic
       (fixed seed), so runs are reproducible. */
    ptiIndex max_dim = 0;
    ptiMatrix **U = malloc((nmodes + 1) * sizeof *U);
    for(ptiIndex m = 0; m < nmodes; ++m) {
        U[m] = malloc(sizeof **U);
        ptiNewMatrix(U[m], X.ndims[m], R);
        ptiRandomizeMatrix(U[m]);
        if(X.ndims[m] > max_dim) max_dim = X.ndims[m];
    }
    U[nmodes] = malloc(sizeof **U);
    ptiNewMatrix(U[nmodes], max_dim, R);
    ptiConstantMatrix(U[nmodes], 0);

    /* The mode order lists the target mode first, then the rest */
    ptiIndex *order = malloc(nmodes * sizeof *order);
    order[0] = mode;
    for(ptiIndex i = 1; i < nmodes; ++i) order[i] = (mode + i) % nmodes;

    /* Sequential */
    if(ptiMTTKRP(&X, U, order, mode) != 0) return 1;
    printf("MTTKRP output is %u x %u; [0][0] = %f\n",
           X.ndims[mode], R, (double) U[nmodes]->values[0]);

#ifdef HIPARTI_USE_OPENMP
    /* OpenMP (last argument: thread count); same result up to float rounding */
    ptiConstantMatrix(U[nmodes], 0);
    if(ptiOmpMTTKRP(&X, U, order, mode, 4) == 0)
        printf("OpenMP MTTKRP [0][0] = %f\n", (double) U[nmodes]->values[0]);
#endif

    for(ptiIndex m = 0; m <= nmodes; ++m) { ptiFreeMatrix(U[m]); free(U[m]); }
    free(U); free(order);
    ptiFreeSparseTensor(&X);
    return 0;
}
