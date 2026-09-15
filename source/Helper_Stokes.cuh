// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

/*! \file Helper_Stokes.cuh
    \brief Declares GPU kernel code for helper functions integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
//! Define the step_one kernel
#ifndef __HELPER_STOKES_CUH__
#define __HELPER_STOKES_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

using namespace hoomd;

__global__ void Stokes_getIndex(
                Scalar4 *d_pos,
		Scalar4 *d_ori,
                const unsigned int* d_body_data,
                const unsigned int* d_rtag,
                int *d_body_tag, //output
                int *d_local_index, //output
                Scalar3 *d_rel_pos,  //output
		Scalar3 *d_rel_pos_bf,
                unsigned int group_size,
                unsigned int *d_group_members,
                const BoxDim box
		);

__global__ void Stokes_update_rel_pos(
                Scalar4 *d_pos,
                const unsigned int* d_rtag,
                int *d_body_tag, //output
                Scalar3 *d_rel_pos,  //output
                unsigned int group_size,
                unsigned int *d_group_members,
                const BoxDim box
                );

__global__ void Stokes_SetForce_kernel(
					Scalar4 *d_net_force,
					Scalar4 *d_net_torque,
					Scalar   *d_AppliedForce,
					unsigned int group_size,
					unsigned int *d_group_members
					);

__global__ void Stokes_SetForce_manually_kernel(
						const Scalar4 *d_pos,     //input
						Scalar   *d_AppliedForce,  //output
						Scalar *d_Velocity,
						unsigned int group_size,
						unsigned int *d_group_members,
						int *d_body_tag,
						const unsigned int *d_nneigh, 
						unsigned int *d_nlist, 
						const size_t *d_headlist,
						const Scalar F_rep,
						const Scalar F_att,
						const Scalar kappa,
						const Scalar k_n,
						const Scalar epsq,
						const Scalar rcut,
						Scalar T_ext,
						Scalar F_ext,
						const BoxDim box,
						const Scalar delta
						);


__global__ void Stokes_SetVelocity_kernel(
						Scalar4 *d_vel,
						Scalar4 *d_omg,
						Scalar   *d_Velocity,
						unsigned int group_size,
						unsigned int *d_group_members
						);


#endif
