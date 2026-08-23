Getting started with HiParTI
===========================


To use HiParTI in a C/C++ project, simply add
```c
#include <HiParTI.h>
```
to your source code.

Link your code with
```sh
-fopenmp -lHiParTI -lm
```

This intro document can help you get used to the basics of HiParTI.


Data types
----------

`ptiValue`: the default real value data type, `float` or `double` depending on
`HIPARTI_VALUE_TYPEWIDTH` (32 by default, i.e. `float`).

`ptiIndex` / `ptiNnzIndex`: index types for mode coordinates and nonzero counts,
`uint32_t` and `uint64_t` by default (`HIPARTI_INDEX_TYPEWIDTH`).

`ptiValueVector`: dense dynamic array of `ptiValue` scalars. It is implemented as a one-dimensional array. It uses preallocation to reduce the overhead of the append operation.

`ptiIndexVector` / `ptiNnzIndexVector`: dense dynamic arrays of `ptiIndex` / `ptiNnzIndex` scalars.

`ptiMatrix`: dense matrix type. It is implemented as a two-dimensional array. Column count is aligned as multiples of 8.

`ptiSparseMatrix`: sparse matrix type in coordinate (COO) storage format. It stores the coordinates and the value of every non-zero entry.

`ptiSparseTensor`: sparse tensor type in coordinate (COO) storage format. It works similar to `ptiSparseMatrix`, but supports tensors with arbitrary modes (number of dimensions).

`ptiSemiSparseTensor`: semi-sparse tensor type in sCOO storage format (details explained in [our SC16-IA3 paper](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf). It can be considered as "a sparse tensor with dense fibers".


Creating objects
----------------

Most data types can fit themselves into stack memory, as local variables. They will handle extra memory allocations on demand.

For example, to construct a `ptiValueVector` and use it.

```c
// Construct it
ptiValueVector my_vector;
ptiNewValueVector(&my_vector, 0, 0);

// Add values to it
ptiAppendValueVector(&my_vector, 42);
ptiAppendValueVector(&my_vector, 31);

// Copy it to another uninitialized vector (last argument: number of threads)
ptiValueVector another_vector;
ptiCopyValueVector(&another_vector, &my_vector, 1);

// Access data
printf("%f %f\n", another_vector.data[0], another_vector.data[1]);

// Free memory
ptiFreeValueVector(&my_vector);
ptiFreeValueVector(&another_vector);
```

A complete first program
------------------------

Save as `demo.c`, then build with
`gcc demo.c -I<HiParTI>/include -L<HiParTI>/build -lHiParTI -fopenmp -lm -o demo`
and run as `./demo <HiParTI>/data/tensors/3d_7.tns`:

```c
#include <HiParTI.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if(argc < 2) { fprintf(stderr, "usage: %s tensor.tns\n", argv[0]); return 1; }

    ptiSparseTensor X;
    if(ptiLoadSparseTensor(&X, 1, argv[1]) != 0) return 1;  // 1 = file is 1-indexed
    ptiSparseTensorStatus(&X, stdout);

    // MTTKRP on mode 0 with random rank-8 factor matrices
    ptiIndex const mode = 0, R = 8;
    ptiIndex nmodes = X.nmodes, max_dim = 0;
    ptiMatrix **U = malloc((nmodes + 1) * sizeof *U);
    for(ptiIndex m = 0; m < nmodes; ++m) {
        U[m] = malloc(sizeof **U);
        ptiNewMatrix(U[m], X.ndims[m], R);
        ptiRandomizeMatrix(U[m]);       // deterministic: same values every run
        if(X.ndims[m] > max_dim) max_dim = X.ndims[m];
    }
    U[nmodes] = malloc(sizeof **U);     // output / scratch matrix
    ptiNewMatrix(U[nmodes], max_dim, R);
    ptiConstantMatrix(U[nmodes], 0);

    ptiIndex order[8];                  // mode order: the target mode first
    order[0] = mode;
    for(ptiIndex i = 1; i < nmodes; ++i) order[i] = (mode + i) % nmodes;

    if(ptiMTTKRP(&X, U, order, mode) != 0) return 1;
    printf("MTTKRP result (first row): %f ...\n", U[nmodes]->values[0]);

    for(ptiIndex m = 0; m <= nmodes; ++m) { ptiFreeMatrix(U[m]); free(U[m]); }
    free(U);
    ptiFreeSparseTensor(&X);
    return 0;
}
```

Most functions require initialized data structures. While functions named `New` or `Copy` require uninitialized data structions. They are states in the Doxygen document on a function basis. Failing to supply data with correct initialization state may result in memory leak or program crash.


Validation
----------

For the sake of simplicity, properties are not designed. You can directly modify any field of any struct.

Every function assumes the input is valid, and guarantees the output is valid. This reduces the the need to check the input for most of the time, and improves the performance as a math library.

But if you modify the data structure directly, you must keep it valid. Some functions expect ordered input, you should sort them with functions like `ptiSparseTensorSortIndex` after your modification, or the functions may not work correctly. These functions usually also produces ordered output.


Error reporting
---------------

Most functions return 0 when it succeeded, non-zero when failed.

By invoking `ptiGetLastError`, you can extract the last error information.

Operating system `errno` and CUDA error code are also captured and converted.

If you need to make sure a procedure produces no error, call `ptiClearLastError` first, since success procedures does not clear last error status automatically.

Limitation: Memory might not be released properly when an error happened. The application will be in an inconsistent state. This might be fixed in future releases.
