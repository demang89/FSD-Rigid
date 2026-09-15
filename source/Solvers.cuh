// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Saddle_Helper.cuh
    \brief Declared functions for saddle point calculations
*/
//! Define the step_one kernel
#ifndef __SOLVERS_CUH__
#define __SOLVERS_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include <stdlib.h>

#include <cusparse.h>
#include <cusolverSp.h>

void Solvers_Saddle(
			Scalar *d_rhs, 
			Scalar *d_solution,
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			Scalar tol,
			void *pBuffer,
			KernelData *ker_data,
			MobilityData *mob_data,
			ResistanceData *res_data,
			WorkData *work_data
			);


#endif
