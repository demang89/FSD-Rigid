// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

/*! \file Helper_Brownian.cuh
    \brief Declares GPU kernel code for helper functions in Brownian calculations.
*/
//! Define the step_one kernel
#ifndef __HELPER_BROWNIAN_CUH__
#define __HELPER_BROWNIAN_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "DataStruct.h"

#include <cufft.h>

__global__ void Brownian_FarField_Dot1of2_kernel(
						Scalar4 *d_a, 
						Scalar4 *d_b, 
						Scalar *dot_sum, 
						unsigned int group_size, 
						unsigned int *d_group_members
						);

__global__ void Brownian_FarField_Dot2of2_kernel(
						Scalar *dot_sum,
						unsigned int num_partial_sums
						);

__global__ void Brownian_FarField_LanczosMatrixMultiply_kernel(
								Scalar4 *d_A,
								Scalar *d_x, 
								Scalar4 *d_b, 
								unsigned int group_size, 
								int m
								);

__global__ void Brownian_NearField_LanczosMatrixMultiply_kernel(
								Scalar *d_A,
								Scalar *d_x,
								Scalar *d_b,
								unsigned int group_size,
								int numel,
								int m
								);

__global__ void Brownian_LanczosMatrixMultiply_kernel(
							Scalar *d_A,
							Scalar *d_x,
							Scalar *d_b,
							unsigned int group_size,
							int stride,
							int m
							);

__global__ void Brownian_FarField_AddGrids_kernel(
						CUFFTCOMPLEX *d_a, 
						CUFFTCOMPLEX *d_b, 
						CUFFTCOMPLEX *d_c, 
						unsigned int NxNyNz
						);

__global__ void Brownian_Farfield_LinearCombinationFTS_kernel(
								Scalar4 *d_a, 
								Scalar4 *d_b, 
								Scalar4 *d_c, 
								Scalar coeff_a, 
								Scalar coeff_b, 
								unsigned int group_size, 
								unsigned int *d_group_members
								);


void Sqrt_multiply( 
			Scalar *d_V,       //input
		    	Scalar *h_alpha,   //input
		    	Scalar *h_beta,    //input
		    	Scalar *h_alpha1,  //input
		    	Scalar *h_beta1,   //input       
		    	int m,            //input                 
		    	Scalar *d_y,       //output
		    	int stride,
		    	int group_size,
		    	KernelData *ker_data,
		    	WorkData *work_data
			);


#endif
