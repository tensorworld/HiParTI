/* Example 3: CP decomposition (CP-ALS).
 *
 *   ./03_cpd ../data/tensors/3D_12031.tns 8        (tensor, rank)
 *
 * Covers: ptiNewKruskalTensor, ptiCpdAls, reading the fit / lambda /
 * factor matrices out of the result.
 */
#include <HiParTI.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if(argc < 3) {
        fprintf(stderr, "usage: %s tensor.tns rank\n", argv[0]);
        return 1;
    }
    ptiIndex const R = (ptiIndex) atoi(argv[2]);

    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, argv[1]) != 0) return 1;

    ptiKruskalTensor K;
    ptiNewKruskalTensor(&K, X.nmodes, X.ndims, R);

    /* 50 iterations max, stop when the fit changes by less than 1e-6.
       fit = 1 - |X - model|/|X|, so 1 is a perfect reconstruction. */
    if(ptiCpdAls(&X, R, 50, 1e-6, &K) != 0) {
        fprintf(stderr, "CPD failed\n");
        return 1;
    }

    printf("\nfinal fit: %f\n", K.fit);
    printf("lambda:");
    for(ptiIndex r = 0; r < R; ++r) printf(" %g", (double) K.lambda[r]);
    printf("\nfactor sizes:");
    for(ptiIndex m = 0; m < X.nmodes; ++m)
        printf(" %ux%u", K.factors[m]->nrows, K.factors[m]->ncols);
    printf("\n");

    ptiFreeKruskalTensor(&K);
    ptiFreeSparseTensor(&X);
    return 0;
}
