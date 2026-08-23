/*
    This file is part of HiParTI!.

    HiParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    HiParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with HiParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * GPU CP-ALS.
 *
 * Mirrors CpdAlsStep in cpd.c, with the MTTKRP - by far the dominant cost -
 * computed on the GPU through ptiCudaMTTKRPOneKernel.  The small dense
 * R x R work per iteration (Gram matrices, the normal-equation solve,
 * normalisation, and the fit) stays on the CPU, exactly as in the OpenMP
 * variant.  Factor initialisation uses the same deterministic
 * ptiRandomizeMatrix as the CPU path, so for a given tensor and rank the
 * CPU and GPU runs follow the same trajectory up to float reordering -
 * which is also what makes this directly testable.
 */

#include <HiParTI.h>
#include <assert.h>
#include <math.h>
#include "clapack.h"
#include "sptensor.h"

static double CudaCpdAlsStep(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  int const impl_num,
  ptiMatrix ** mats,  // Row-major
  ptiValue * const lambda)
{
  ptiIndex const nmodes = ptien->nmodes;
  ptiIndex const stride = mats[0]->stride;
  double fit = 0;

  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(ptien->ndims[m] == mats[m]->nrows);
    ptiAssert(mats[m]->ncols == rank);
  }

  ptiValue alpha = 1.0, beta = 0.0;
  char notrans = 'N';
  char uplo = 'L';
  int blas_rank = (int) rank;
  int blas_stride = (int) stride;

  ptiMatrix * tmp_mat = mats[nmodes];
  ptiMatrix ** ata = (ptiMatrix **)malloc((nmodes+1) * sizeof(*ata));
  for(ptiIndex m=0; m < nmodes+1; ++m) {
    ata[m] = (ptiMatrix *)malloc(sizeof(ptiMatrix));
    ptiAssert(ptiNewMatrix(ata[m], rank, rank) == 0);
    ptiAssert(mats[m]->stride == ata[m]->stride);
  }

  /* Compute all "ata"s: ata[m] = mats[m]^T * mats[m] (upper triangle) */
  for(ptiIndex m=0; m < nmodes; ++m) {
    int blas_nrows = (int)(mats[m]->nrows);
    ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
      mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
  }

  double oldfit = 0;
  ptiIndex * mats_order = (ptiIndex*)malloc(nmodes * sizeof(*mats_order));

  for(ptiIndex it=0; it < niters; ++it) {
    ptiTimer timer;
    ptiNewTimer(&timer, 0);
    ptiStartTimer(timer);

    for(ptiIndex m=0; m < nmodes; ++m) {
      tmp_mat->nrows = mats[m]->nrows;

      /* Factor Matrices order */
      mats_order[0] = m;
      for(ptiIndex i=1; i<nmodes; ++i)
          mats_order[i] = (m+i) % nmodes;

      /* GPU MTTKRP; the result lands in mats[nmodes] on the host,
         the same contract as ptiMTTKRP. */
      ptiAssert (ptiCudaMTTKRPOneKernel(ptien, mats, mats_order, m, impl_num) == 0);

      memcpy(mats[m]->values, tmp_mat->values, mats[m]->nrows * stride * sizeof(ptiValue));

      /* Solve ata[nmodes] * X = mats[m] */
      ptiAssert ( ptiMatrixSolveNormals(m, nmodes, ata, mats[m]) == 0 );

      /* Normalise mats[m]; store the norms in lambda */
      if (it == 0 ) {
        ptiMatrix2Norm(mats[m], lambda);
      } else {
        ptiMatrixMaxNorm(mats[m], lambda);
      }

      /* ata[m] = mats[m]^T * mats[m] */
      int blas_nrows = (int)(mats[m]->nrows);
      ssyrk_(&uplo, &notrans, &blas_rank, &blas_nrows, &alpha,
        mats[m]->values, &blas_stride, &beta, ata[m]->values, &blas_stride);
    } // Loop nmodes

    fit = KruskalTensorFit(ptien, lambda, mats, ata);

    ptiStopTimer(timer);
    double its_time = ptiElapsedTime(timer);
    ptiFreeTimer(timer);

    printf("  its = %" HIPARTI_PRI_INDEX " ( %.3lf s ) fit = %0.5f  delta = %+0.4e\n",
        it+1, its_time, fit, fit - oldfit);
    if(it > 0 && fabs(fit - oldfit) < tol) {
      break;
    }
    oldfit = fit;
  } // Loop niters

  GetFinalLambda(rank, nmodes, mats, lambda);

  for(ptiIndex m=0; m < nmodes+1; ++m) {
    ptiFreeMatrix(ata[m]);
    free(ata[m]);
  }
  free(ata);
  free(mats_order);

  return fit;
}


int ptiCudaCpdAls(
  ptiSparseTensor const * const ptien,
  ptiIndex const rank,
  ptiIndex const niters,
  double const tol,
  int const impl_num,
  ptiKruskalTensor * ktensor)
{
  ptiIndex nmodes = ptien->nmodes;

  /* The CUDA MTTKRP kernels are 3-D only, and only these launch geometries
     exist; fail loudly rather than compute nothing. */
  if(nmodes != 3) {
    fprintf(stderr, "[CUDA SpTns CPD-ALS] Error: only 3-D tensors are supported on GPU (got %u modes).\n", nmodes);
    return -1;
  }
  if(impl_num != 11 && impl_num != 15 && impl_num != 16) {
    fprintf(stderr, "[CUDA SpTns CPD-ALS] Error: unsupported impl_num %d (valid: 11,15,16).\n", impl_num);
    return -1;
  }

  /* Initialise factor matrices, identically to ptiCpdAls */
  ptiIndex max_dim = ptiMaxIndexArray(ptien->ndims, nmodes);
  ptiMatrix ** mats = (ptiMatrix **)malloc((nmodes+1) * sizeof(*mats));
  for(ptiIndex m=0; m < nmodes+1; ++m) {
    mats[m] = (ptiMatrix *)malloc(sizeof(*mats[m]));
  }
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiAssert(ptiNewMatrix(mats[m], ptien->ndims[m], rank) == 0);
    ptiAssert(ptiRandomizeMatrix(mats[m]) == 0);
  }
  ptiAssert(ptiNewMatrix(mats[nmodes], max_dim, rank) == 0);
  ptiAssert(ptiConstantMatrix(mats[nmodes], 0) == 0);

  ptiTimer timer;
  ptiNewTimer(&timer, 0);
  ptiStartTimer(timer);

  ktensor->fit = CudaCpdAlsStep(ptien, rank, niters, tol, impl_num, mats, ktensor->lambda);

  ptiStopTimer(timer);
  ptiPrintElapsedTime(timer, "GPU  SpTns CPD-ALS");
  ptiFreeTimer(timer);

  ktensor->factors = mats;
  /* mats[nmodes] is scratch, not a factor; ptiFreeKruskalTensor frees only
     the first nmodes entries (same as ptiCpdAls). */
  ptiFreeMatrix(mats[nmodes]);
  free(mats[nmodes]);

  return 0;
}
