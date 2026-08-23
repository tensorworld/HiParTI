HiParTI examples
================

Three small, heavily commented programs to read alongside
[GettingStarted.md](../GettingStarted.md).  They are built automatically with
the library; after `./build.sh` you will find them in `build/examples/`.

| Program | What it shows |
| --- | --- |
| `01_load_query` | Loading a `.tns` file, reading the tensor's fields, dumping it back out |
| `02_mttkrp` | Setting up factor matrices and running MTTKRP (sequential and OpenMP) |
| `03_cpd` | CP decomposition with `ptiCpdAls`, and reading fit / lambda / factors |

Run them from the repository root, e.g.:

```sh
./build/examples/01_load_query data/tensors/3d_7.tns
./build/examples/02_mttkrp     data/tensors/3d_7.tns 0 8
./build/examples/03_cpd        data/tensors/3D_12031.tns 8
```

To build one outside this tree (against an installed HiParTI):

```sh
gcc 02_mttkrp.c $(pkg-config --cflags --libs hiparti) -o 02_mttkrp
```
