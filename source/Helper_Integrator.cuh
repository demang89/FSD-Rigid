// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Integrator.cuh
    \brief Declares helper functions for integration.
*/
//! Define the step_one kernel
#ifndef __HELPER_INTEGRATOR_CUH__
#define __HELPER_INTEGRATOR_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

using namespace hoomd;

__global__ void Integrator_RFD_RandDisp_kernel(
						Scalar *d_psi,
						unsigned int N,
						int stride,
						const uint64_t timestep,
						const unsigned int seed
						);

__global__ void Integrator_ZeroVelocity_kernel( 
						Scalar *d_b,
						unsigned int N
						);
__global__ void Integrator_AddStrainRate_kernel( 
						Scalar *d_b,
						Scalar shear_rate,
						Scalar4 *d_pos,
						unsigned int *d_group_members,
						// float B2,
						// float *d_sqm_B2_mask,
						// Scalar4 *d_ori,						
						unsigned int N,
						Scalar3 *d_rel_pos
						);

#endif
