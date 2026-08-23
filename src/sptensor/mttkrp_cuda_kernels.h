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

#ifndef PARTI_MTTKRP_KERNELS_H
#define PARTI_MTTKRP_KERNELS_H

__device__ void lock(int* mutex);
__device__ void unlock(int* mutex);


/* impl_num = 01 */
/* impl_num = 02 */
/* impl_num = 03 */
/* impl_num = 04 */
/* impl_num = 05 */
/* impl_num = 06 */
/* impl_num = 09, for arbitraty nmodes. Scratch is necessary for tensors with arbitrary modes. */
/**** impl_num = 1x: One GPU using one kernel ****/
/* impl_num = 11 */
__global__ void pti_MTTKRPKernelNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 12 */
__global__ void pti_MTTKRPKernelRankNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 15 */
__global__ void pti_MTTKRPKernelRankSplitNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

/* impl_num = 16 */
__global__ void pti_MTTKRPKernelRankSplitNnzRB3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);



/**** impl_num = 2x: Stream One GPU: cache blocking ****/
/* impl_num = 21. */
/* impl_num = 25 */
/* impl_num = 26 */
/**** impl_num = 3x: Stream One GPU: shared memory blocking for coarse grain ****/
/* impl_num = 35 */
/* impl_num = 36 */
/**** impl_num = 4x: Stream One GPU: shared memory blocking for medium grain ****/
/* impl_num = 45 */
/* impl_num = 46 */
/**** impl_num = 5x: multiple GPUs ****/
/* impl_num = 59, only the interface is a bit different. */
/* impl_num = 31 */
__global__ void pti_MTTKRPKernelNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

__global__ void pti_MTTKRPKernelRankSplitNnz3DOneKernel(
    const ptiIndex mode,
    const ptiIndex nmodes,
    const ptiNnzIndex nnz,
    const ptiIndex R,
    const ptiIndex stride,
    const ptiIndex * Xndims,
    ptiIndex ** const Xinds,
    const ptiValue * Xvals,
    const ptiIndex * dev_mats_order,
    ptiValue ** dev_mats);

#endif