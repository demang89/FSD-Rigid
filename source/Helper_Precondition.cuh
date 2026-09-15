// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Precondition.cuh
    \brief Declares helper functions for error checking and sparse math.
*/
//! Define the step_one kernel
#ifndef __HELPER_PRECONDITION_CUH__
#define __HELPER_PRECONDITION_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include <cusparse.h>

using namespace hoomd;

__global__ void Precondition_ZeroVector_kernel( 
						Scalar *d_b,
						const unsigned int nnz,
						const unsigned int group_size
						);

__global__ void Precondition_ApplyRCM_Vector_kernel( 
						Scalar *d_Scratch_Vector,
						Scalar *d_Vector,
						const int *d_prcm,
						const int length,
						const int direction
						);

__global__ void Precondition_AddInt_kernel(
					unsigned int *d_a,
					unsigned int *d_b,
					unsigned int *d_c,
					int coeff_a,
					int coeff_b,
					unsigned int group_size 
					);

__global__ void Precondition_AddIdentity_kernel(
						Scalar *d_L_Val,
						int   *d_L_RowPtr,
						int   *d_L_ColInd, 
						int group_size,
						Scalar ichol_relaxer
						);

__global__ void Precondition_Inn_kernel(
					Scalar *d_y,
					Scalar *d_x,
					int *d_HasNeigh,
					int group_size
					);

__global__ void Precondition_ImInn_kernel(
					Scalar *d_y,
					Scalar *d_x,
					int *d_HasNeigh,
					int group_size
					);


__global__ void Precondition_ExpandPRCM_kernel(
						int *d_prcm,
						int *d_scratch,
						int group_size
						);

__global__ void Precondition_InitializeMap_kernel(
						int *d_map,
						int nnz
						);

__global__ void Precondition_Map_kernel(
					Scalar *d_Scratch,
					Scalar *d_Val,
					int *d_map,
					int nnz
					);

__global__ void Precondition_GetDiags_kernel(
						int group_size, 
						Scalar *d_Diag,
						int   *d_L_RowPtr,
						int   *d_L_ColInd,
						Scalar *d_L_Val
						);

__global__ void Precondition_DiagMult_kernel(
						Scalar *d_y, // output
						Scalar *d_x, // input
						int group_size, 
						Scalar *d_Diag,
						int direction
						);

__global__ void Precondition_ZeroUpperTriangle_kernel( 
							int *d_RowPtr,
							int *d_ColInd,
							Scalar *d_Val,
							int group_size
							);

__global__ void Precondition_Lmult_kernel( 
					Scalar *d_y,
					Scalar *d_x,
					int *d_RowPtr,
					int *d_ColInd,
					Scalar *d_Val,
					int group_size
					);

__global__ void buildTransformationCSR(
    					int* row_ptr, 
					int* col_index, 
					Scalar* values, 
					int *body_tag,
					Scalar3* rel_pos, 
					int Np, 
					unsigned int *d_group_members
					);
	
#endif
