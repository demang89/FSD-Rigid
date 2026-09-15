// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Brownian_NearField.cuh
    \brief Declares GPU kernel code for Near-Field Brownian Calculation
*/
//! Define the kernel
#ifndef __BROWNIAN_NEARFIELD_CUH__
#define __BROWNIAN_NEARFIELD_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "Helper_Saddle.cuh"
#include <cufft.h>

#include "DataStruct.h"

#include <cusparse.h>
#include <cusolverSp.h>

__global__ void Brownian_NearField_RNG_kernel(
						Scalar *d_Psi_nf,
						unsigned int N,
						const unsigned int seed,
						const uint64_t timestep
						//const float T,
						//const float dt
						);


void Brownian_NearField_Force(
				Scalar *d_FBnf, // output
				Scalar4 *d_pos,
				unsigned int *d_group_members,
				unsigned int group_size,
				const BoxDim& box,
				Scalar dt,
				void *pBuffer,
				KernelData *ker_data,
				BrownianData *bro_data,
				ResistanceData *res_data,
				WorkData *work_data,
				uint64_t timestep
				);

#endif
