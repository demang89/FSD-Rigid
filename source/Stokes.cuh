/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander
// Modified by Gang Wang
// Modified by Andrew Fiore
// Modified by Zhouyang Ge

/*! \file Stokes.cuh
    \brief Declares GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.
*/
//! Define the step_one kernel
#ifndef __STOKES_CUH__
#define __STOKES_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#include <cufft.h>

#include "DataStruct.h"
#include "ContactFriction.cuh"

#include <cusparse.h>
#include <cusolverSp.h>


//! Kernel driver for the first part (no second part) of the Stokes update called by Stokes.cc
cudaError_t Stokes_StepOne( uint64_t timestep,
				unsigned int output_period,
				Scalar4 *d_pos,
				Scalar4 *d_ori,
				Scalar4 *d_vel,
				Scalar4 *d_angmom,
				Scalar4 *d_net_force,
				Scalar4 *d_net_torque,
				Scalar *d_AppliedForce,
				Scalar *d_Velocity,
				Scalar *d_stress,
				Scalar T_ext,
				Scalar F_ext,
				Scalar dt,
				Scalar shear_rate,
				const Scalar delta,
				unsigned int block_size,
				int3 *d_image,
				unsigned int *d_group_members,
				unsigned int *d_body_data,
				unsigned int group_size,
				const BoxDim& box,
				BrownianData *bro_data,
				MobilityData *mob_data,
				ResistanceData *res_data,
				WorkData *work_data,
				const bool fric
				);




cudaError_t Stokes_StepTwo(     uint64_t timestep,
				Scalar4 *d_pos,
				Scalar4 *d_net_force,
				Scalar4 *d_buf,         // input (buffered position) / output (HOOMD velocity)
				float *d_AppliedForce,
				float *d_Velocity,
				Scalar dt,
				float shear_rate,
				unsigned int block_size,
				int3 *d_image,
				unsigned int *d_group_members,
				unsigned int group_size,
				const BoxDim& box,
				BrownianData *bro_data,
				MobilityData *mob_data,
				ResistanceData *res_data,
				WorkData *work_data
				);


void setupContactTable(ContactSlot *d_table, unsigned int group_size, unsigned int max_contact);

#endif
