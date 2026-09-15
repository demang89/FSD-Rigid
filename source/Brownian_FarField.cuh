// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Brownian_FarField.cuh
    \brief Declares GPU kernel code for far-field Brownian Calculation.
*/
//! Define the kernel
#ifndef __BROWNIAN_FARFIELD_CUH__
#define __BROWNIAN_FARFIELD_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include "DataStruct.h"


void Brownian_FarField_SlipVelocity(
			        	Scalar *d_Uslip_ff,
					Scalar4 *d_pos,
					int *d_local_index,
					unsigned int *d_group_members,
					unsigned int group_size,
					const BoxDim& box,
					Scalar dt,
			        	BrownianData *bro_data,
			        	MobilityData *mob_data,
					KernelData *ker_data,
					WorkData *work_data,
					uint64_t timestep
					);

#endif
