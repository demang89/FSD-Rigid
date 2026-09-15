// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Saddle.cuh
    \brief Declared helper functions for saddle point calculations
*/
//! Define the step_one kernel
#ifndef __HELPER_SADDLE_CUH__
#define __HELPER_SADDLE_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include <stdlib.h>
#include "cusparse.h"

using namespace hoomd;

__global__ void Saddle_ZeroOutput_kernel( 
					Scalar *d_b, 
					unsigned int N 
					);

__global__ void Saddle_AddFloat_kernel( 
					Scalar *d_a, 
					Scalar *d_b,
					Scalar *d_c,
					Scalar coeff_a,
					Scalar coeff_b,
					unsigned int N,
					int stride
					);

__global__ void Saddle_SplitGeneralizedF_kernel( 	
						Scalar *d_GeneralF, 
						Scalar4 *d_net_force,
						Scalar4 *d_TorqueStress,
						unsigned int N
						);

__global__ void Saddle_MakeGeneralizedU_kernel( 	
						Scalar *d_GeneralU, 
						Scalar4 *d_vel,
						Scalar4 *d_AngvelStrain,
						unsigned int N
						);


//Deepak:added for rigid
__global__ void sigma_kernel(   
				Scalar *d_a,
                                Scalar *d_b,
                                Scalar3 *d_c,
                                int *d_body_tag,
                                unsigned int *d_group_members_p,
                                unsigned int group_size_p
                                );


__global__ void sigma_transpose_kernel( 
					Scalar *d_a,
                                        Scalar *d_b,
                                        Scalar3 *d_c,
                                        int *d_body_tag,
                                        unsigned int *d_group_members_p,
                                        unsigned int group_size_p
                                        );


__global__ void subtract_uinf_kernel( 
					Scalar *d_a,
                                        Scalar4 *d_pos,
                                        unsigned int *d_group_members,
                                        unsigned int group_size,
                                        Scalar shear_rate
                                        );
#endif
