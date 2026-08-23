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
#include <HiParTI.h>
#include "sptensor.h"
#include <cuda_runtime.h>


template <typename T>
__device__ static void print_array(const T array[], ptiNnzIndex length, T start_index) {
    if(length == 0) {
        return;
    }
    printf("%d", (int) (array[0] + start_index));
    ptiNnzIndex i;
    for(i = 1; i < length; ++i) {
        printf(", %d", (int) (array[i] + start_index));
    }
    printf("\n");
}


__device__ static void print_array(const ptiValue array[], ptiNnzIndex length, ptiNnzIndex start_index) {
    if(length == 0) {
        return;
    }
    printf("%.2f", array[0] + start_index);
    ptiNnzIndex i;
    for(i = 1; i < length; ++i) {
        printf(", %.2f", array[i] + start_index);
    }
    printf("\n");
}


__device__ void lock(int* mutex) {
  /* compare mutex to 0.
     when it equals 0, set it to 1
     we will break out of the loop after mutex gets set to  */
    while (atomicCAS(mutex, 0, 1) != 0) {
    /* do nothing */
    }
}


__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
}





/* impl_num = 01 */
/* impl_num = 02 */
/* impl_num = 03 */
/* impl_num = 04 */
/* impl_num = 05 */
/* impl_num = 06 */
/* impl_num = 09 */
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
    ptiValue ** dev_mats)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }


    const ptiNnzIndex tidx = threadIdx.x;
    ptiNnzIndex x;

    ptiIndex const * const mode_ind = Xinds[mode];
    ptiValue * const mvals = (ptiValue*)dev_mats[nmodes];
    ptiIndex times_mat_index = dev_mats_order[1];
    ptiValue * times_mat = dev_mats[times_mat_index];
    ptiIndex * times_inds = Xinds[times_mat_index];
    ptiIndex times_mat_index_2 = dev_mats_order[2];
    ptiValue * times_mat_2 = dev_mats[times_mat_index_2];
    ptiIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
              ptiIndex const mode_i = mode_ind[x];
              ptiIndex tmp_i = times_inds[x];
              ptiValue const entry = Xvals[x];
              ptiIndex tmp_i_2 = times_inds_2[x];
              ptiValue tmp_val = 0;
              for(ptiIndex r=0; r<R; ++r) {
              tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
              atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }  

}



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
    ptiValue ** dev_mats)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.x;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;  // index rank
    const ptiNnzIndex tidy = threadIdx.y;  // index nnz
    ptiNnzIndex x;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;


    ptiIndex const * const mode_ind = Xinds[mode];
    ptiValue * const mvals = (ptiValue*)dev_mats[nmodes];
    ptiIndex times_mat_index = dev_mats_order[1];
    ptiValue * times_mat = dev_mats[times_mat_index];
    ptiIndex * times_inds = Xinds[times_mat_index];
    ptiIndex times_mat_index_2 = dev_mats_order[2];
    ptiValue * times_mat_2 = dev_mats[times_mat_index_2];
    ptiIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.x + tidx + nl * nnz_per_loop;
        if(x < nnz) {
            ptiIndex const mode_i = mode_ind[x];
            ptiIndex tmp_i = times_inds[x];
            ptiValue const entry = Xvals[x];
            ptiIndex tmp_i_2 = times_inds_2[x];
            ptiValue tmp_val = 0;
            ptiIndex r;

            for(ptiIndex l=0; l<num_loops_r; ++l) {
                r = tidy + l * blockDim.y;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidy + num_loops_r * blockDim.y;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
   
    }

}



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
    ptiValue ** dev_mats)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;  // index rank
    const ptiNnzIndex tidy = threadIdx.y;  // index nnz
    ptiNnzIndex x;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;


    ptiIndex const * const mode_ind = Xinds[mode];
    ptiValue * const mvals = (ptiValue*)dev_mats[nmodes];
    ptiIndex times_mat_index = dev_mats_order[1];
    ptiValue * times_mat = dev_mats[times_mat_index];
    ptiIndex * times_inds = Xinds[times_mat_index];
    ptiIndex times_mat_index_2 = dev_mats_order[2];
    ptiValue * times_mat_2 = dev_mats[times_mat_index_2];
    ptiIndex * times_inds_2 = Xinds[times_mat_index_2];

    for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
        x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
        if(x < nnz) {
            ptiIndex const mode_i = mode_ind[x];
            ptiIndex tmp_i = times_inds[x];
            ptiValue const entry = Xvals[x];
            ptiIndex tmp_i_2 = times_inds_2[x];
            ptiValue tmp_val = 0;
            ptiIndex r;

            for(ptiIndex l=0; l<num_loops_r; ++l) {
                r = tidx + l * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }

            if(rest_loop > 0 && tidx < rest_loop) {
                r = tidx + num_loops_r * blockDim.x;
                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
   
    }

}


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
    ptiValue ** dev_mats)
{
    ptiNnzIndex num_loops_nnz = 1;
    ptiNnzIndex const nnz_per_loop = gridDim.x * blockDim.y;
    if(nnz > nnz_per_loop) {
        num_loops_nnz = (nnz + nnz_per_loop - 1) / nnz_per_loop;
    }

    const ptiNnzIndex tidx = threadIdx.x;  // index rank
    const ptiNnzIndex tidy = threadIdx.y;  // index nnz
    ptiNnzIndex x;
    const ptiIndex num_loops_r = R / blockDim.x;
    const ptiIndex rest_loop = R - num_loops_r * blockDim.x;
    ptiIndex r;


    ptiIndex const * const mode_ind = Xinds[mode];
    ptiValue * const mvals = (ptiValue*)dev_mats[nmodes];
    ptiIndex times_mat_index = dev_mats_order[1];
    ptiValue * times_mat = dev_mats[times_mat_index];
    ptiIndex * times_inds = Xinds[times_mat_index];
    ptiIndex times_mat_index_2 = dev_mats_order[2];
    ptiValue * times_mat_2 = dev_mats[times_mat_index_2];
    ptiIndex * times_inds_2 = Xinds[times_mat_index_2];


    for(ptiIndex l=0; l<num_loops_r; ++l) {
        r = tidx + l * blockDim.x;

        for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                ptiIndex const mode_i = mode_ind[x];
                ptiIndex tmp_i = times_inds[x];
                ptiValue const entry = Xvals[x];
                ptiIndex tmp_i_2 = times_inds_2[x];
                ptiValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }  // End for l: num_loops_r

    if(rest_loop > 0 && tidx < rest_loop) {
        r = tidx + num_loops_r * blockDim.x;

        for(ptiNnzIndex nl=0; nl<num_loops_nnz; ++nl) {
            x = blockIdx.x * blockDim.y + tidy + nl * nnz_per_loop;
            if(x < nnz) {
                ptiIndex const mode_i = mode_ind[x];
                ptiIndex tmp_i = times_inds[x];
                ptiValue const entry = Xvals[x];
                ptiIndex tmp_i_2 = times_inds_2[x];
                ptiValue tmp_val = 0;

                tmp_val = entry * times_mat[tmp_i * stride + r] * times_mat_2[tmp_i_2 * stride + r];
                atomicAdd(&(mvals[mode_i * stride + r]), tmp_val);
            }
        }
    }   // End if rest_loop

}



/* impl_num = 21. */
/* impl_num = 25 */
/* impl_num = 26 */
/* impl_num = 35 */
/* impl_num = 36 */
/* impl_num = 45 */
/* impl_num = 46 */
/* impl_num = 59 */
