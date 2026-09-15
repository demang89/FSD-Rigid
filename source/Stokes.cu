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

#include "Stokes.cuh"

#include "Integrator.cuh"
#include "Lubrication.cuh"
#include "Mobility.cuh"
#include "Precondition.cuh"
#include "Wrappers.cuh"
#include "Saddle.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Stokes.cuh"

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>

#include "hoomd/TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! \file Stokes.cu
    \brief Defines GPU kernel code for integration considering hydrodynamic interactions on the GPU. Used by Stokes.cc.
*/

/*! 
	Step one of two-step integrator (step 2 is null) for the overdamped particle dynamics.
	Explicit Euler integration of particle positions given a velocity.

	timestep        (input)         current timestep
	output_period   (input)         output per output_period steps
	d_pos		(input/ouput)	array of particle positions
	d_ori		(input/ouput)	array of particle orientations
	d_net_force	(input)		particle forces
	d_vel		(output)	particle velocities
	d_AppliedForce	(input/output)	Array for force and torque applied on particles
	d_Velocity	(input/output)	Array for linear and angular velocity of particles and stresslets
	dt		(input)		integration time step
	m_error		(input)		calculation error tolerance
	shear_rate	(input)		shear rate in the suspension, if any
	block_size	(input)		number of threads per block for particle-based calculations
	d_image		(input)		array of particle images
	d_group_members	(input)		index of particles within the integration group
	group_size	(input)		number of particles
	box		(input)		periodic box information
	bro_data	(input)		structure containing data for Brownian calculations
	mob_data	(input)		structure containing data for Mobility calculations
	res_data	(input)		structure containing data for lubrication resistance calculations
	work_data	(input)		structure containing data for scratch arrays and workspaces
*/

cudaError_t Stokes_StepOne(     uint64_t timestep,
				unsigned int output_period,
				Scalar4 *d_pos,
				Scalar4 *d_ori,  
				Scalar4 *d_vel,
				Scalar4 *d_angmom,
				Scalar4 *d_net_force,
				Scalar4 *d_net_torque,
				Scalar *d_AppliedForce,
				Scalar *d_Velocity,
				Scalar *d_Stress,
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
				const bool m_fric
				){

	// *******************************************************
	// Pre-calculation setup
	// *******************************************************
	
	// Set up the blocks and threads to run the particle-based kernels
	int gridBlockSize = ( group_size > block_size ) ? block_size : group_size;
	dim3 grid_p(((group_size+gridBlockSize-1)/gridBlockSize), 1, 1 );
	dim3 threads_p(gridBlockSize, 1, 1);

	gridBlockSize = ( res_data->N_rigid > block_size ) ? block_size : res_data->N_rigid;
	dim3 grid_b((res_data->N_rigid/gridBlockSize) + 1, 1, 1 );
	dim3 threads_b(gridBlockSize, 1, 1);

	// Set up the blocks and threads to run the FFT-grid-based kernels	
	unsigned int NxNyNz = (mob_data->Nx) * (mob_data->Ny) * (mob_data->Nz);
	gridBlockSize = ( NxNyNz > block_size ) ? block_size : NxNyNz;
	int gridNBlock = ( NxNyNz + gridBlockSize - 1 ) / gridBlockSize ; 

	// Initialize values in the data structure for kernel information
	KernelData ker_struct = {grid_p,
				 threads_p,
				 grid_b,
				 threads_b,
				 gridNBlock,
				 gridBlockSize,
				 NxNyNz};
	KernelData *ker_data = &ker_struct;

	// *******************************************************
        // Get sheared grid vectors
	// *******************************************************

	Mobility_SetGridk_kernel<<<gridNBlock,gridBlockSize>>>(mob_data->gridk,  //output
								mob_data->Nx,	  
								mob_data->Ny,	  
								mob_data->Nz,	  
								NxNyNz,		  
								box,		  
								mob_data->xi,	  
								mob_data->eta);

	//Deepak:get particle bodies and local indexes
	Stokes_getIndex<<< grid_p, threads_p >>>(
						d_pos,
						d_ori,
						d_body_data,
						res_data->d_rtag,
						res_data->body_tag,
						res_data->local_index,
						res_data->rel_pos,
						res_data->rel_pos_bf,
						group_size,
						d_group_members,
						box
						);

	//Deepak : check if there is any lubrication interaction
	int* d_has_lubrication;
	cudaMalloc(&d_has_lubrication, sizeof(int));
 	cudaMemset(d_has_lubrication, 0, sizeof(int));
	check_Lubrication_interactions<<< grid_p, threads_p >>>( d_pos, 
							res_data->body_tag, 
							d_group_members, 
							group_size,
							box,
							res_data->nneigh,
							res_data->nlist,
							res_data->headlist,
							res_data->rlub,
							d_has_lubrication);
	cudaMemcpy(&(res_data->has_lubrication), d_has_lubrication, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("lubrication is working? %d \n",res_data->has_lubrication); //debug
	cudaFree(d_has_lubrication);
	d_has_lubrication = NULL;

	// *******************************************************
        // Prepare the preconditioners
	// *******************************************************
	
	// Build preconditioner (only do once, because it should still be
	// sufficiently good for RFD with small displacements)
	// zhoge: It mainly does the incomplete Cholesky factorization of P * (\tilde{R}_FU^nf + relaxer*I) * P^T
	Precondition_Wrap(d_pos,           
			res_data->d_rtag, 
			box,		   
			ker_data,	   
			res_data,
			work_data);

	
	//Debug_Lattice_SpinViscosity(mob_data,res_data,ker_data,work_data,d_pos,d_group_members,group_size,box);
	//Debug_Lattice_ShearViscosity(mob_data,res_data,ker_data,work_data,d_pos,d_group_members,group_size,box);
	//Debug_Lattice_ExtensionalViscosity(mob_data,res_data,ker_data,work_data,d_pos,d_group_members,group_size,box);
	//gpuErrchk(cudaPeekAtLastError());
	//return cudaSuccess;

	// *******************************************************
        // Solve the hydrodynamic problem and do the integration
	// *******************************************************
	
	// Set applied force equal to net_force from HOOMD (pair potentials, external potentials, etc.)
	unsigned int group_size_b = res_data->N_rigid;
	cudaMemset(d_AppliedForce, 0, 6 * group_size * sizeof(Scalar));
	cudaMemset(d_Velocity, 0, (12 * group_size_b) * sizeof(Scalar));
	cudaMemset(d_Stress, 0, (22 * group_size) * sizeof(Scalar));
	Stokes_SetForce_kernel<<<grid_p,threads_p>>>( d_net_force, d_net_torque, d_AppliedForce, group_size, d_group_members );
	Stokes_SetForce_manually_kernel<<<grid_p,threads_p>>>(
							d_pos,           //input
							d_AppliedForce,  //output
							&d_Stress[15*group_size],
							group_size,
							d_group_members,
							res_data->body_tag,
							res_data->nneigh, 
							res_data->nlist, 
							res_data->headlist,
							res_data->m_F_rep,
							res_data->m_F_att,
							res_data->m_kappa,
							res_data->m_k_n, 
							res_data->m_epsq,
							res_data->m_rcut,
							T_ext,
							F_ext,
							box,
							delta
							);
							
		
        if(m_fric){
                gpu_ContactFriction_RFU(
                                        d_AppliedForce,
                                        &d_Stress[15*group_size],
                                        d_pos,
                                        d_vel,
                                        d_angmom,
                                        d_group_members,
                                        group_size,
					res_data->body_tag,
					res_data->rel_pos,
                                        box,
                                        res_data->nneigh,
                                        res_data->nlist,
                                        res_data->headlist,
                                        res_data->m_contact_table,
                                        res_data->max_contact,
                                        timestep,
                                        res_data->m_k_t,
                                        res_data->m_k_n,
                                        res_data->m_muf,
                                        res_data->m_F_att,
                                        res_data->m_epsq,
                                        shear_rate,
                                        dt,
					1.0,
                                        ker_data,
                                        res_data->d_tag,
					res_data->d_rtag
                                        );
        }

	// Compute particle velocities from central RFD + Saddle point solve (in Integrator.cu)

        // Allocate the buffer space    
        void *pBuffer;
        cudaMalloc( (void**)&pBuffer, res_data->pBufferSize );//zhoge: pBufferSize computed in Precondition_IChol()

        if ( bro_data->T > 0){
                Integrator_Fixman( d_Velocity,
				d_Stress,
                                d_pos,
				d_ori,
                                d_image,  //input (won't be modified)
                                d_group_members,
                                group_size,
                                box,
                                pBuffer,
                                ker_data,
                                bro_data,
                                mob_data,
                                res_data,
                                work_data,
                                timestep,
                                dt
                                );
        }

	Integrator_ComputeVelocity( timestep,
					output_period,
					d_AppliedForce,
					&d_Velocity[6*group_size_b],      //output (FSD velocity and stresslet, 11N)
					&d_Stress[5*group_size],
					dt,
					shear_rate,
					d_pos,      //input position
					d_image,
					d_group_members,
					group_size,
					box,
					pBuffer,
					ker_data,
					bro_data,
					mob_data,
					res_data,
					work_data
					);

	Integrator_ExplicitEuler_Shear_kernel<<<grid_b,threads_b>>>(d_pos,     //overwrite
								d_ori,         //overwrite	
								d_vel,         //overwrite
								d_angmom,      //overwrite
								d_Velocity, 
								d_image,       //overwrite
								res_data->d_rtag,
								group_size_b,
								box,
								dt,
								shear_rate
								);

	//Clean up
	cudaFree(pBuffer);

	// Error checking
	gpuErrchk(cudaPeekAtLastError());
	return cudaSuccess;
}

void setupContactTable(ContactSlot *d_table, unsigned int group_size, unsigned int max_contact)
{

    // 2. Configure 1D execution configuration for the initialization pass
    unsigned int  total_slots = group_size * max_contact;
    unsigned int threads_per_block = 256;
    dim3 grid(((total_slots + threads_per_block - 1)/threads_per_block), 1, 1 );
    dim3 threads(threads_per_block, 1, 1);

    // 3. Launch initialization kernel
    initContactTable_kernel<<<grid, threads>>>(d_table, total_slots);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in table initialization: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to make sure it's ready before the main simulation loop starts
    cudaDeviceSynchronize();
}
