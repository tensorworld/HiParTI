/* Example 1: load a sparse tensor and look around.
 *
 *   ./01_load_query ../data/tensors/3d_7.tns
 *
 * Covers: ptiLoadSparseTensor, the ptiSparseTensor fields, and dumping.
 */
#include <HiParTI.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    if(argc < 2) {
        fprintf(stderr, "usage: %s tensor.tns\n", argv[0]);
        return 1;
    }

    /* Second argument: the index base used in the FILE (1 for the tensors
       shipped in data/; indices are stored 0-based internally either way).
       TensorSuite-format files are detected and handled automatically. */
    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, argv[1]) != 0) {
        fprintf(stderr, "could not load %s\n", argv[1]);
        return 1;
    }

    /* A one-paragraph summary: dimensions, nnz, density, storage size */
    ptiSparseTensorStatus(&X, stdout);

    /* The struct is open: nmodes, ndims[], nnz, inds[mode].data[z], values.data[z] */
    double sum = 0;
    for(ptiNnzIndex z = 0; z < X.nnz; ++z)
        sum += (double) X.values.data[z];
    printf("sum of all %llu values: %g\n", (unsigned long long) X.nnz, sum);

    printf("first entry: (");
    for(ptiIndex m = 0; m < X.nmodes; ++m)
        printf("%u%s", X.inds[m].data[0], m + 1 < X.nmodes ? ", " : "");
    printf(") = %g\n", (double) X.values.data[0]);

    /* Write it back out (second argument: index base to use in the file) */
    FILE *f = fopen("copy.tns", "w");
    if(f) {
        ptiDumpSparseTensor(&X, 1, f);
        fclose(f);
        printf("wrote copy.tns\n");
    }

    ptiFreeSparseTensor(&X);
    return 0;
}
