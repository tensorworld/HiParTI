HiParTI
------

[![CI](https://github.com/tensorworld/HiParTI/actions/workflows/ci.yml/badge.svg)](https://github.com/tensorworld/HiParTI/actions/workflows/ci.yml)

A Hierarchical Parallel Tensor Infrastructure (HiParTI), is to support fast essential sparse tensor operations and tensor decompositions on multicore CPU and GPU architectures. These basic tensor operations are critical to the overall performance of tensor analysis algorithms (such as tensor decomposition). HiParTI is based on [ParTI!](https://github.com/hpcgarage/ParTI) library developed at GaTech. 


# Contents:

## Supported Data
* General sparse tensors/matrices
* Semi-sparse tensors/matrices with dense dimensions

## Sparse tensor representations:
* Coordinate (COO) format
* Hierarchical Coordinate (HiCOO) format (Refer to [[SC'19 paper]](http://fruitfly1026.github.io/static/files/sc18-li.pdf))
* Semi-COO (sCOO) format (Refer to [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf))
* Semi-HiCOO (sHiCOO) format (Refer to [[pdf]](http://fruitfly1026.github.io/static/files/iiswc20-li.pdf))

## Sparse tensor operations:

* Scala-tensor mul/div
* Element-wise tensor add/sub/mul/div
* Sparse tensor-times-dense vector (SpTTV)
* Sparse tensor-times-dense matrix (SpTTM)
* Sparse matricized tensor times Khatri-Rao product (SpMTTKRP)
* Sparse tensor sorting
* Sparse tensor reordering
* Sparse tensor matricization
* Kronecker product
* Khatri-Rao product

## Sparse tensor decompositions:

* Sparse CANDECOMP/PARAFAC (CP) decomposition, on CPU (sequential and OpenMP) and GPU

# Build requirements:

- C Compiler (GCC or Clang)

- [CUDA SDK](https://developer.nvidia.com/cuda-downloads) (Optional)

- [CMake](https://cmake.org) (>v3.0)

- BLAS and LAPACK. On Ubuntu/Debian: `sudo apt install libblas-dev liblapack-dev`;
  alternatives such as [OpenBLAS](http://www.openblas.net), Intel MKL, or
  [MAGMA](http://icl.cs.utk.edu/magma/) also work (see `build-sample.config`)


# Build:

1. Type `./build.sh` (on the first run it creates `build.config` from
   `build-sample.config`; edit that file to change compilers or options,
   and please leave `-DUSE_OPENMP=ON`)

2. Check `build` for the resulting library

3. Check `build/benchmark` for benchmark programs, and `build/examples` for
   the small commented example programs from [`examples/`](examples/)

For NVIDIA GPU support, set in `build.config`:

```
-DUSE_CUDA=ON
-DCUDA_ARCH_BIN=86        # your GPU's compute capability, e.g. 86 for RTX 30xx/A4500, 80 for A100
```

# Run the tests:

```
cd build && ctest
```

`ctest -L unit` runs only the fast unit tests; with a CUDA build, `ctest -L gpu`
selects the GPU tests (they are skipped automatically when no device is present).

# Install:

```
cd build && sudo make install
```

installs the library, the headers and a pkg-config file (default prefix
`/usr/local`; override with `-DCMAKE_INSTALL_PREFIX=...` in `build.config`).
Your own programs then build with:

```
gcc myprog.c $(pkg-config --cflags --libs hiparti) -o myprog
```

# Data formats:

Tensors are read from text `.tns` files (first line: number of modes; second
line: the dimensions; then one `i j k value` entry per line, 1-indexed - see
`data/tensors/`), and matrices from MatrixMarket `.mtx` files (see
`data/matrices/`).  Files from the
[TensorSuite](https://tensorworld.github.io/TensorSuite/) collection
(`%%TensorSuite-TNS` header, optional value column, sidecar metadata) are
detected and loaded automatically by `ptiLoadSparseTensor`.


# Build docs:

1. Install Doxygen

2. Go to `docs`

3. Type `make`


<br/>The algorithms and details are described in the following publications.
# Publications
* **Performance Implication of Tensor Irregularity and Optimization for Distributed Tensor Decomposition**. Zheng Miao, Jon Calhoun, Rong Ge, Jiajia Li. ACM Transactions on Parallel Computing (TOPC). 2023. [[paper]](https://doi.org/10.1145/3580315)
* **BALA-CPD: BALanced and Asynchronous Distributed Tensor Decomposition**. Zheng Miao, Jiajia Li, Jon Calhoun, Rong Ge. IEEE Cluster. 2022. [[paper]](https://ieeexplore.ieee.org/document/9912661/)
* **Athena: High-Performance Sparse Tensor Contraction Sequence on Heterogeneous Memory**. Jiawen Liu, Dong Li, Roberto Gioiosa, Jiajia Li. International Conference on Supercomputing (ICS). 2021. [[paper]](https://fruitfly1026.github.io/static/files/athena.pdf) [[bib]](https://fruitfly1026.github.io/static/files/athena-bib.txt)
* **Sparta: High-Performance, Element-Wise Sparse Tensor Contraction on Heterogeneous Memory**. Jiawen Liu, Jie Ren, Roberto Gioiosa, Dong Li, Jiajia Li. Principles and Practice of Parallel Programming (PPoPP). 2021.
* **Sparsity-Aware Distributed Tensor Decomposition**. Zheng Miao, Jon C. Calhoun, Rong Ge, Jiajia Li. ACM/IEEE International Conference for High-Performance Computing, Networking, Storage, and Analysis (SC). 2020. (Poster)
* **Efficient and Effective Sparse Tensor Reordering**. Jiajia Li, Bora Ucar, Umit Catalyurek, Kevin Barker, Richard Vuduc. International Conference on Supercomputing (ICS). 2019.
* **HiCOO: Hierarchical Storage of Sparse Tensors**. Jiajia Li, Jimeng Sun, Richard Vuduc. ACM/IEEE International Conference for High-Performance Computing, Networking, Storage, and Analysis (SC). 2018. (Best Student Paper Award) [[pdf]](http://fruitfly1026.github.io/static/files/sc18-li.pdf)
* **Optimizing Sparse Tensor Times Matrix on GPUs**. Yuchen Ma, Jiajia Li, Xiaolong Wu, Chenggang Yan, Jimeng Sun, Richard Vuduc. Journal of Parallel and Distributed Computing (Special Issue on Systems for Learning, Inferencing, and Discovering). 2018.
* **Optimizing Sparse Tensor Times Matrix on multi-core and many-core architectures**. Jiajia Li, Yuchen Ma, Chenggang Yan, Richard Vuduc. The sixth Workshop on Irregular Applications: Architectures and Algorithms (IA^3), co-located with SC’16. 2016. [[pdf]](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf)
* **ParTI!: a Parallel Tensor Infrastructure for Data Analysis**. Jiajia Li, Yuchen Ma, Chenggang Yan, Jimeng Sun, Richard Vuduc. Tensor-Learn Workshop @ NIPS'16. [[pdf]](http://fruitfly1026.github.io/static/files/nips16-tensorlearn.pdf)


# Contributors

* Jiajia Li (Contact: jiajia.li@ncsu.edu or fruitfly1026@gmail.com)
