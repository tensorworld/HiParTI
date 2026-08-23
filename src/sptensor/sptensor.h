/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HIPARTI_SPTENSOR_H
#define HIPARTI_SPTENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <HiParTI.h>
 

double pti_SparseTensorNorm(const ptiSparseTensor *X);

int pti_SparseTensorCompareIndices(ptiSparseTensor * const tsr1, ptiNnzIndex loc1,  ptiSparseTensor * const tsr2, ptiNnzIndex loc2);
int pti_SparseTensorCompareIndicesMorton2D(
    ptiSparseTensor * const tsr1,
    uint64_t loc1, 
    ptiSparseTensor * const tsr2,
    uint64_t loc2,
    ptiIndex * mode_order,
    ptiElementIndex sb_bits);
int pti_SparseTensorCompareIndicesExceptSingleMode(ptiSparseTensor * const tsr1, ptiNnzIndex loc1, ptiSparseTensor * const tsr2, ptiNnzIndex loc2, ptiIndex * const mode_order);
int pti_SparseTensorCompareIndicesExceptSingleModeCantor(ptiSparseTensor * const tsr1, ptiNnzIndex loc1, ptiSparseTensor * const tsr2, ptiNnzIndex loc2, ptiIndex * const mode_order);
int pti_SparseTensorCompareIndicesRowBlock(
    ptiSparseTensor * const tsr1,
    ptiNnzIndex loc1,
    ptiSparseTensor * const tsr2,
    ptiNnzIndex loc2,
    ptiElementIndex sk_bits);
int pti_SparseTensorCompareIndicesExceptSingleModeRowBlock(
    ptiSparseTensor * const tsr1,
    ptiNnzIndex loc1,
    ptiSparseTensor * const tsr2,
    ptiNnzIndex loc2,
    ptiIndex * const mode_order,
    ptiElementIndex sk_bits);
int pti_SparseTensorCompareIndicesRange(ptiSparseTensor * const tsr, ptiNnzIndex loc, ptiIndex * const inds1, ptiIndex * const inds2);
int pti_SparseTensorCompareIndicesCustomize(ptiSparseTensor * const tsr1, ptiNnzIndex loc1, ptiIndex * const mode_order_1, ptiSparseTensor * const tsr2, ptiNnzIndex loc2, ptiIndex * const mode_order_2, ptiIndex num_ncmodes);
void pti_SwapValues(ptiSparseTensor *tsr, ptiNnzIndex ind1, ptiNnzIndex ind2);

void pti_SparseTensorCollectZeros(ptiSparseTensor *tsr);

int pti_DistSparseTensor(ptiSparseTensor * tsr,
    int const nthreads,
    ptiNnzIndex * const dist_nnzs,
    ptiIndex * dist_nrows);

int pti_DistSparseTensorFixed(ptiSparseTensor * tsr,
    int const nthreads,
    ptiNnzIndex * const dist_nnzs,
    ptiNnzIndex * dist_nrows);

int pti_GetSubSparseTensor(ptiSparseTensor *dest, const ptiSparseTensor *tsr, const ptiIndex limit_low[], const ptiIndex limit_high[]);




#ifdef __cplusplus
}
#endif

#endif
