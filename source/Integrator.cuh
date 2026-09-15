// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

/*! \file Integrator.cuh
    \brief Declares GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
//! Define the kernel
#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include <cufft.h>

#include "DataStruct.h"

#include <cusparse.h>
#include <cusolverSp.h>

extern "C" __global__ void Integrator_ExplicitEuler_kernel(
                                                                Scalar4 *d_pos_in,
                                                                Scalar4 *d_pos_out,
                                                                Scalar4 *d_ori,
                                                                Scalar *d_Velocity,
                                                                int3 *d_image,
                                                                unsigned int *d_rtag,
                                                                Scalar3 *d_rel_pos_bf,
                                                                unsigned int N_agg,
                                                                unsigned int nb,
                                                                BoxDim box,
                                                                Scalar dt,
                                                                Scalar shear_rate
                                                                );

extern "C" __global__ void Integrator_ExplicitEuler_Shear_kernel(
                                                                 Scalar4 *d_pos,
                                                                 Scalar4 *d_ori,
                                                                 Scalar4 *d_vel,
                                                                 Scalar4 *d_angmom,
                                                                 Scalar *d_Velocity,
                                                                 int3 *d_image,
                                                                 unsigned int *d_group_members,
                                                                 unsigned int group_size,
                                                                 BoxDim box,
                                                                 Scalar dt,
                                                                 Scalar shear_rate
                                                                 );


void Integrator_Fixman( Scalar *d_Velocity, //size 11N, but only the first 6N are modified
                     Scalar *d_Stress,
                     Scalar4 *d_pos,
                     Scalar4 *d_ori,
                     int3 *d_image,
                     unsigned int *d_group_members,
                     unsigned int group_size,
                     const BoxDim& box,
                     void *pBuffer,
                     KernelData *ker_data,
                     BrownianData *bro_data,
                     MobilityData *mob_data,
                     ResistanceData *res_data,
                     WorkData *work_data,
                     const uint64_t timestep,
                     Scalar dt
                     );


void Integrator_ComputeVelocity( uint64_t timestep,
				     unsigned int output_period,
				     Scalar *d_AppliedForce,
				     Scalar *d_Velocity,
				     Scalar *d_Stress,
				     Scalar dt,
				     Scalar shear_rate,
				     Scalar4 *d_pos,
				     int3 *d_image,
				     unsigned int *d_group_members,
				     unsigned int group_size,
				     const BoxDim& box,
				     void *pBuffer,
				     KernelData *ker_data,
				     BrownianData *bro_data,
				     MobilityData *mob_data,
				     ResistanceData *res_data,
				     WorkData *work_data
				     );

__global__ void Integrator_UEinf_kernel(
                                        Scalar *d_a,
                                        const Scalar shear_rate,
                                        const unsigned int *d_group_members,
                                        unsigned int group_size,
                                        const Scalar3 *d_rel_pos
                                        );

__global__ void AddBeadStress_kernel(
                Scalar *d_y,
                const Scalar *d_x,
                const Scalar3 *rel_pos,
                unsigned int group_size,
                unsigned int *d_group_members
                );

__global__ void print_forces(Scalar *d_force, Scalar4 *d_ori, unsigned int *d_group_members, unsigned int group_size, unsigned int *d_tag);
#endif
