// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Modified by Zhouyang Ge

#include "Integrator.cuh"

#include "Brownian_FarField.cuh"
#include "Brownian_NearField.cuh"
#include "Lubrication.cuh"
#include "Mobility.cuh"
#include "Precondition.cuh"
#include "Solvers.cuh"
#include "Wrappers.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Integrator.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Precondition.cuh"
#include "Helper_Saddle.cuh"
#include "Helper_Stokes.cuh"

#include <curand.h>
#include <cuda_runtime.h>

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>
#include <math.h>

#include "lapacke.h"
#include "cublas_wrappers.hpp"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

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
                                                                ){

        // Thread ID
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;

        if(tidx>=N_agg) return;

        // Particle ID
        unsigned int idx = d_rtag[tidx];

        // read the particle's posision
        Scalar4 pos4 = d_pos_in[idx];
        Scalar3 pos = make_scalar3(pos4.x, pos4.y, pos4.z);

        // read the particle's velocity and update position
        Scalar ux = d_Velocity[ 6*tidx ];
        Scalar uy = d_Velocity[ 6*tidx + 1 ];
        Scalar uz = d_Velocity[ 6*tidx + 2 ];
        Scalar wx = d_Velocity[ 6*tidx + 3 ];
        Scalar wy = d_Velocity[ 6*tidx + 4 ];
        Scalar wz = d_Velocity[ 6*tidx + 5 ];

        Scalar3 vel = make_scalar3( ux, uy, uz);
        Scalar3 omg = make_scalar3( wx, wy, wz);

        // Add the shear
        vel.x += shear_rate * pos.y;
        omg.z -= shear_rate/2.;
        //Deepak:modified rotational part
        quat<Scalar> q(d_ori[idx]);
        q += Scalar(0.5) * dt * vec3<Scalar>(omg) * q;
        q = q * (Scalar(1.0) / sqrt(norm2(q)));

        // Update the positions 
        Scalar3 dx = vel * dt;
        pos += dx;

        // Read in particle's image and wrap periodic boundary
        int3 image = d_image[idx];
        box.wrap(pos, image);

        // write out the results
        d_pos_out[idx] = make_scalar4(pos.x, pos.y, pos.z, pos4.w);

        int start = nb * tidx + N_agg;
        for (int s = 0; s < nb; ++s) {
                int sphere_i = d_rtag[ start+s ];
                Scalar4 posP =  d_pos_in[sphere_i];

                vec3<Scalar> local_pos(d_rel_pos_bf[sphere_i]);
                Scalar3 dr = vec_to_scalar3(rotate(q,local_pos));
                Scalar3 posi = pos + dr;
                int3 image = d_image[sphere_i];
                box.wrap(posi, image);
                d_pos_out[sphere_i] = make_scalar4(posi.x, posi.y, posi.z, posP.w);
        }
}



__global__ void AddBeadStress_kernel(
                Scalar *d_y,
                const Scalar *d_x,
                const Scalar3 *rel_pos,
                unsigned int group_size,
                unsigned int *d_group_members
                ){
        // Thread index
        unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        // Check if thread is inbounds
        if ( tidx < group_size ) {
                // Particle ID
                unsigned int idx = d_group_members[tidx];
		Scalar pressure = (rel_pos[idx].x * d_x[6*tidx+0] + rel_pos[idx].y * d_x[6*tidx+1] + rel_pos[idx].z * d_x[6*tidx+2])/3.0;
                d_y[5*tidx+0] += (rel_pos[idx].x * d_x[6*tidx+0] - pressure);
                d_y[5*tidx+1] += 0.5*(rel_pos[idx].x * d_x[6*tidx+1] + rel_pos[idx].y * d_x[6*tidx+0]);
                d_y[5*tidx+2] += 0.5*(rel_pos[idx].x * d_x[6*tidx+2] + rel_pos[idx].z * d_x[6*tidx+0]);
                d_y[5*tidx+3] += 0.5*(rel_pos[idx].y * d_x[6*tidx+2] + rel_pos[idx].z * d_x[6*tidx+1]);
                d_y[5*tidx+4] += (rel_pos[idx].y * d_x[6*tidx+1] - pressure);
        }
}


__global__ void Integrator_UEinf_kernel(
					Scalar *d_a,
					const Scalar shear_rate, 
					const unsigned int *d_group_members,
					unsigned int group_size,
					const Scalar3 *d_rel_pos
					){
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        if(tidx<group_size){
		int idx = d_group_members[tidx];
		d_a[6*tidx]   = 0.5 * shear_rate * d_rel_pos[idx].y;
		d_a[6*tidx+1] = 0.5 * shear_rate * d_rel_pos[idx].x;
	}
}


/*! \file Integrator.cu
    \brief Defines integrator functions to capture Brownian drift in the
		velocity and stresslet. 
*/


/*! 
	Integrates particle position according to the Explicit Euler scheme, with shear
   
	d_pos_in		(input)  3Nx1 particle positions at initial point
	d_pos_out		(output) 3Nx1 new particle positions
	d_ori                   (input/output) 4Nx1 particle orientations
	d_Velocity		(input)  6Nx1 generalized particle velocities
	d_image			(input)  particle periodic images
	d_group_members		(input)  indices of the mebers of the group to integrate
	group_size		(input)  Number of members in the group
	box Box			(input)  dimensions for periodic boundary condition handling
	dt			(input)  timestep
	shear_rate		(input)  shear rate for the system
	
	This kernel must be executed with a 1D grid of any block size such that the number of threads is greater than or
	equal to the number of members in the group. The kernel's implementation simply reads one particle in each thread
	and updates that particle. 
*/

extern "C" __global__ void Integrator_ExplicitEuler_Shear_kernel(Scalar4 *d_pos,
                                                                 Scalar4 *d_ori,
                                                                 Scalar4 *d_vel,
                                                                 Scalar4 *d_angmom,
                                                                 Scalar *d_Velocity,
                                                                 int3 *d_image,
                                                                 unsigned int *d_rtag,
                                                                 unsigned int group_size,
                                                                 BoxDim box,
                                                                 Scalar dt,
                                                                 Scalar shear_rate
                                                                 ){

        // Thread ID
        int tidx = blockIdx.x * blockDim.x + threadIdx.x;

        // Check that thread is in bounds
        if ( tidx < group_size ){

                //max fluid velocity
                Scalar vinf = shear_rate * box.getL().y;

                // Particle ID
                unsigned int idx = d_rtag[tidx];

                // read the particle's posision
                Scalar4 pos4 = d_pos[idx];
                Scalar3 pos = make_scalar3(pos4.x, pos4.y, pos4.z);

                // read the particle's velocities
                Scalar ux = d_Velocity[ 6*tidx     ] + d_Velocity[ 6*group_size + 6*tidx     ];
                Scalar uy = d_Velocity[ 6*tidx + 1 ] + d_Velocity[ 6*group_size + 6*tidx  + 1];
                Scalar uz = d_Velocity[ 6*tidx + 2 ] + d_Velocity[ 6*group_size + 6*tidx  + 2];
                Scalar wx = d_Velocity[ 6*tidx + 3 ] + d_Velocity[ 6*group_size + 6*tidx  + 3];
                Scalar wy = d_Velocity[ 6*tidx + 4 ] + d_Velocity[ 6*group_size + 6*tidx  + 4];
                Scalar wz = d_Velocity[ 6*tidx + 5 ] + d_Velocity[ 6*group_size + 6*tidx  + 5];

                Scalar3 vel = make_scalar3( ux, uy, uz);
                Scalar3 omg = make_scalar3( wx, wy, wz);

                // Add the shear
                vel.x += shear_rate * pos.y;
                omg.z -= shear_rate/2.;

                //Deepak:modified rotational part
                quat<Scalar> q(d_ori[idx]);
                q += Scalar(0.5) * dt * vec3<Scalar>(omg) * q;
                q = q * (Scalar(1.0) / sqrt(norm2(q)));
                d_ori[idx] = quat_to_scalar4(q);

                // Update the positions 
                Scalar3 dx = vel * dt;
                pos += dx;
                int3 image = d_image[idx];
                int img_y = image.y;
                box.wrap(pos, image);
                img_y -= image.y;

                d_pos[idx] = make_scalar4(pos.x,pos.y,pos.z,pos4.w);
                d_image[idx] = image;

                //update velocity and angular velocity
                d_vel[idx] = make_scalar4(vel.x+vinf*img_y, vel.y, vel.z, d_vel[idx].w);
                d_angmom[idx] = make_scalar4(omg.x, omg.y, omg.z, d_angmom[idx].w);

                // quat<Scalar> p = Scalar(2.0) * q * vec3<Scalar>(omg);
                // d_angmom[idx] = quat_to_scalar4(p);
        }
}


/*! 
        Random Finite Differencing to compute the divergence of inverse(RFU)

        zhoge: Now only update the first 6N entries corresponding to the Brownian drift.
               The associated stress is computed together with the rest of velocities.
               Note that, since we use RSU^nf later, the stress might be slightly off. (To check)

        d_Divergence            (output) 11Nx1 divergence of RFU (first 6N) and RSU (last 5N)
        d_pos                   (input)  particle positions
        d_image                 (input)  particle periodic image
        d_group_members         (input)  ID of particle within integration group
        group_size              (input)  number of particles
        box                     (input)  periodic box information
        ker_data                (input)  structure containing kernel launch information
        bro_data                (input)  structure containing Brownian calculation information
        mob_data                (input)  structure containing mobility calculation information
        res_data                (input)  structure containing lubrication calculation information

*/
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
                     )
{
        cublasHandle_t blasHandle = work_data->blasHandle;

        // Get kernel information
        dim3 grid    = ker_data->particle_grid;
        dim3 threads = ker_data->particle_threads;

        dim3 grid_b    = ker_data->rigid_grid;
        dim3 threads_b = ker_data->rigid_threads;

        Scalar n = bro_data->rfd_epsilon;

        int group_size_b = res_data->N_rigid;
        unsigned int numel = 11*group_size + 6*group_size_b;

        // Random Vectors
        Scalar4 *d_posPrime = work_data->saddle_posPrime;

        // RHS and Solution Vectors
        Scalar *d_rhs = bro_data->rfd_rhs;
        Scalar *d_sol = bro_data->rfd_sol;

        Scalar *d_rhs_temp = res_data->Scratch5;

        // Zero out the rhs and solution vectors
        cudaMemset(d_rhs, 0, numel * sizeof(Scalar));
        cudaMemset(d_sol, 0, numel * sizeof(Scalar));
        cudaMemset(d_rhs_temp, 0, 6 * group_size * sizeof(Scalar));

        Brownian_FarField_SlipVelocity( d_rhs, //output
                                        d_pos,
                                        res_data->local_index,
                                        d_group_members,
                                        group_size,
                                        box,
                                        dt,
                                        bro_data,
                                        mob_data,
                                        ker_data,
                                        work_data,
                                        timestep
                                        );

        // Compute the near-field Brownian (lubrication) force/torque and add (or minus, doesn't matter) to the RHS
        // d_rhs[11N:17N] = F_B^nf
        if(res_data->has_lubrication)
        {
                Brownian_NearField_Force( d_rhs_temp, //output
                                        d_pos,
                                        d_group_members,
                                        group_size,
                                        box,
                                        dt,
                                        pBuffer,
                                        ker_data,
                                        bro_data,
                                        res_data,
                                        work_data,
                                        timestep
                                        );
        }
        //projecting d_rhs[11N:17N] to cluster-level
        sigma_kernel<<< grid, threads >>>(d_rhs_temp, &d_rhs[11*group_size],
                                                        res_data->rel_pos, res_data->body_tag, d_group_members, group_size);

        // Solve the saddle point problem
        Solvers_Saddle( d_rhs,
                        d_sol,  //output (contains U+)
                        d_pos,
                        d_group_members,
                        group_size,
                        box,
                        1e-6,
                        pBuffer,
                        ker_data,
                        mob_data,
                        res_data,
                        work_data
                        );

        //Copy random velocity and far-field stresslet
        cudaMemcpy( d_Velocity, &d_sol[11*group_size], 6*group_size_b*sizeof(Scalar), cudaMemcpyDeviceToDevice );
        cudaMemcpy( d_Stress, &d_sol[6*group_size], 5*group_size*sizeof(Scalar), cudaMemcpyDeviceToDevice );
        Scalar isqdt = sqrt(1.0/dt);
        // Multiply it (d_Velocity[0:6*group_size_b+5*group_size]) with sqrt(1/dt)
        cublas::scal<Scalar>( blasHandle, 6*group_size_b, &isqdt, d_Velocity, 1 );
        cublas::scal<Scalar>( blasHandle, 5*group_size, &isqdt, d_Stress, 1 );

        // Compute the near-field hydrodynamic stresslet and force(RFU.U)
        Scalar *d_vel = res_data->EinfForce;
        sigma_transpose_kernel<<< grid, threads >>>(d_Velocity, d_vel,
                                                res_data->rel_pos, res_data->body_tag, d_group_members, group_size);
        if(res_data->has_lubrication){
                Lubrication_RSU_kernel<<< grid, threads >>>(
                                                        d_Stress, // output
                                                        d_vel, // input
                                                        d_pos,
                                                        res_data->body_tag,
                                                        res_data->local_index,
                                                        d_group_members,
                                                        group_size,
                                                        box,
                                                        res_data->nneigh,
                                                        res_data->nlist,
                                                        res_data->headlist,
                                                        res_data->table_dist,
                                                        res_data->table_vals,
                                                        res_data->table_min,
                                                        res_data->table_dr,
                                                        res_data->rlub
                                                        );
        }

        // Do the displacements (zhoge: no need to update quaterians since this is a fake update)
        Integrator_ExplicitEuler_kernel<<<grid_b,threads_b>>>(
                                                                d_pos,
                                                                d_posPrime,
                                                                d_ori,
                                                                d_Velocity,
                                                                d_image,
                                                                res_data->d_rtag,
                                                                res_data->rel_pos_bf,
                                                                res_data->N_rigid,
                                                                res_data->m_nb,
                                                                box,
                                                                dt/n,
                                                                0
                                                                );

        Stokes_update_rel_pos<<< grid, threads >>>(
                                                d_posPrime,
                                                res_data->d_rtag,
                                                res_data->body_tag,
                                                res_data->rel_pos,
                                                group_size,
                                                d_group_members,
                                                box
                                                );

        // Solve the saddle point problem
        Solvers_Saddle( d_rhs,
                        d_sol,  //output (contains U+)
                        d_posPrime,
                        d_group_members,
                        group_size,
                        box,
                        1e-6,
                        pBuffer,
                        ker_data,
                        mob_data,
                        res_data,
                        work_data );

  // Compute the near-field hydrodynamic stresslet
        cublas::scal<Scalar>( blasHandle, 6*group_size_b+11*group_size, &isqdt, d_sol, 1 );
        sigma_transpose_kernel<<< grid, threads >>>(&d_sol[11*group_size], d_vel,
                                                res_data->rel_pos, res_data->body_tag, d_group_members, group_size);
        if(res_data->has_lubrication){
                Lubrication_RSU_kernel<<< grid, threads >>>(
                                                        &d_sol[6*group_size], // output
                                                        d_vel, // input
                                                        d_posPrime,
                                                        res_data->body_tag,
                                                        res_data->local_index,
                                                        d_group_members,
                                                        group_size,
                                                        box,
                                                        res_data->nneigh,
                                                        res_data->nlist,
                                                        res_data->headlist,
                                                        res_data->table_dist,
                                                        res_data->table_vals,
                                                        res_data->table_min,
                                                        res_data->table_dr,
                                                        res_data->rlub
                                                        );
        }

        Scalar fac = n/2.0;
        Saddle_AddFloat_kernel<<<grid,threads>>>( d_Velocity, &d_sol[11*group_size], &d_sol[11*group_size],-fac, fac, group_size_b, 6 );
        Saddle_AddFloat_kernel<<<grid,threads>>>( d_Velocity, &d_sol[11*group_size], d_Velocity, 1.0, 1.0, group_size_b, 6 );
        Saddle_AddFloat_kernel<<<grid,threads>>>( d_Stress, &d_sol[6*group_size], d_Stress, fac, -fac, group_size, 5 );

        Stokes_update_rel_pos<<< grid, threads >>>(
                                                d_pos,
                                                res_data->d_rtag,
                                                res_data->body_tag,
                                                res_data->rel_pos,
                                                group_size,
                                                d_group_members,
                                                box
                                                );
        // Clean up
        d_sol = NULL;
        d_rhs = NULL;
        d_posPrime = NULL;
        d_vel = NULL;
        d_rhs_temp = NULL;
}


/*! 
	Combine all the parts required to compute the particle displacements

	timestep                (input)  current timestep
	output_period           (input)  output per output_period steps
	d_AppliedForce		(input)  6Nx1 particle generalized forces
	d_Velocity		(output) 11Nx1 particle generalized velocities (6N) and stresslets (5N)
	dt			(input)  integration timestep
	shear_rate		(input)	 shear rate for imposed shear flow
	d_pos			(input)  particle positions
	sqm_B2                  (input)  B2 mode coef (spherical squirmers)
	d_ori                   (input)  particle orientations
	d_image			(input)  particle periodic image
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	mob_data		(input)  structure containing mobility calculation information
	res_data		(input)  structure containing lubrication calculation information


*/
void Integrator_ComputeVelocity(    uint64_t timestep,
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
				){
	// cuBLAS handle
	cublasHandle_t blasHandle = work_data->blasHandle;

	int group_size_b = res_data->N_rigid;
		
	// Dereference kernel data for grid and threads
	dim3 grid    = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Set up RHS and solution vectors
	Scalar *d_rhs      = work_data->saddle_rhs;
	Scalar *d_solution = work_data->saddle_solution;  //keep it as the initial guess for the current saddle (boyuan)

	// Zero RHS and velocity to start
	Scalar scale = 0.0;  
	cudaMemset(d_rhs, 0, (11 * group_size + 6 * group_size_b) * sizeof(Scalar));
	//cudaMemset(d_solution, 0, (11 * group_size + 6 * group_size_b) * sizeof(Scalar));
	//if((timestep%100)==0) cudaMemset(d_solution, 0, (11 * group_size + 6 * group_size_b) * sizeof(Scalar));

        Scalar *d_force = res_data->Scratch5;
        cudaMemset(d_force, 0, 6 * group_size * sizeof(Scalar));

	// Add (E_inf - E_s) to the RHS, modifying d_rhs[6N:11N]
	Integrator_AddStrainRate_kernel<<< grid, threads >>>( d_rhs,
								shear_rate,     
								d_pos,
								d_group_members,
								group_size,
								res_data->rel_pos
								); 


	//temporary pointer to store near-field and direct forces on primary particles
	Scalar *d_rhs_temp = res_data->Scratch4;
	cudaMemset(d_rhs_temp, 0, 6 * group_size * sizeof(Scalar));

	//Deepak:added for U_Einf
	Scalar *d_EinfForce = res_data->EinfForce;
	if(res_data->has_lubrication){

		// Add -RFU.UFE to the RHS, modifying d_rhs[11N:17N]
		cudaMemset(d_EinfForce, 0, 6 * group_size * sizeof(Scalar));

		//Compute UFE=E.(x-X)
		Integrator_UEinf_kernel<<<grid,threads>>>(
							d_EinfForce,
							shear_rate,
							d_group_members,
							group_size,
							res_data->rel_pos
							);
		//Compute RFU.UFE						
		Lubrication_RFU_kernel<<<grid,threads>>>(
							d_rhs_temp, // output
							d_EinfForce, // input (relative velocity)
							d_pos,
							res_data->body_tag,
							res_data->local_index,
							d_group_members,
							group_size,
							box,
							res_data->nneigh,
							res_data->nlist,
							res_data->headlist,
							res_data->table_dist,
							res_data->table_vals,
							res_data->table_min,
							res_data->table_dr,
							res_data->rlub);
		//Compute -RFU.UFE
		scale = -1.0;
		cublas::scal<Scalar>( blasHandle, 6*group_size, &scale, d_rhs_temp, 1 );


		// Add -R_FE^nf:E_inf
        	Lubrication_RFE_kernel<<< grid, threads >>>( d_rhs_temp,   //output
        	                                           shear_rate,
        	                                           d_pos,
        	                                           res_data->body_tag,
        	                                           d_group_members,
        	                                           group_size,
        	                                           box,
        	                                           res_data->nneigh,
        	                                           res_data->nlist,
        	                                           res_data->headlist,
        	                                           res_data->table_dist,
        	                                           res_data->table_vals,
        	                                           res_data->table_min,
        	                                           res_data->table_dr,
        	                                           res_data->rlub );
  	}

	// add VDW+contact force to the total force on the beads
	scale = -1.0;
	cublas::axpy<Scalar>( blasHandle, 6*group_size, &scale, d_AppliedForce, 1, d_rhs_temp, 1 );
        Saddle_AddFloat_kernel<<<grid,threads>>>(d_force, d_rhs_temp, d_force, 0, -1.0, group_size, 6 );

	//map the bead's force to the aggregate force
	sigma_kernel<<< grid, threads >>>(d_rhs_temp, &d_rhs[11*group_size],
					res_data->rel_pos, res_data->body_tag, d_group_members, group_size);

	// Do the saddle point solve (GMRES)
	Solvers_Saddle( d_rhs, 
			d_solution,  //output (first 6N: ff force/torque, next 5N: ff stress, last 6N: relative velocities)
			d_pos,
			d_group_members,
			group_size,
			box,
			bro_data->tol,
			pBuffer,
			ker_data,
			mob_data,
			res_data,
			work_data
			);
		
	// Get velocity and far-field hydrodynamic stresslet out of solution vector
	scale = 1.0;
	cublas::axpy<Scalar>( blasHandle, 6*group_size_b, &scale, &d_solution[11*group_size], 1, d_Velocity, 1 );
	cublas::axpy<Scalar>( blasHandle, 5*group_size, &scale, &d_solution[6*group_size], 1, d_Stress, 1 );
	
	// Only process stresslets if they are to be written to output files
	if ( ( output_period > 0 ) && ( int(timestep+1) % output_period == 0 ) ) 
    	{
		//Add far-field hydrodynamic force to the total force on the beads
		Saddle_AddFloat_kernel<<<grid,threads>>>( d_solution, d_force, d_force, 1.0, 1.0, group_size, 6 );

		// Add the near-field contributions to the stresslet
		// - RSU_nf * (U-Uinf)
		if(res_data->has_lubrication){
                        //Map aggregate velocity to the bead's velocity
			cudaMemset(d_rhs_temp, 0, 6 * group_size * sizeof(Scalar));
			sigma_transpose_kernel<<< grid, threads >>>(&d_solution[11*group_size], d_rhs_temp,
								res_data->rel_pos, res_data->body_tag, d_group_members, group_size);

                        //Compute near-field hydrodynamic force on beads
			Lubrication_RFU_kernel<<<grid,threads>>>(
								d_rhs, // output
								d_rhs_temp, // input (relative velocity)
								d_pos,
								res_data->body_tag,
								res_data->local_index,
								d_group_members,
								group_size, 
								box,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist, 
								res_data->table_dist,
								res_data->table_vals,
								res_data->table_min,
								res_data->table_dr,
								res_data->rlub
								);

                        //Add near-field HI to the total force on the bead
			Saddle_AddFloat_kernel<<<grid,threads>>>( d_rhs, d_force, d_force, -1.0, 1.0, group_size, 6 );

                        //Compute near-field hydrodynamic stresslet
			scale = -1.0;
			cublas::axpy<Scalar>( blasHandle, 6*group_size, &scale, d_EinfForce, 1, d_rhs_temp, 1 );

                        // - RSU_nf : (U-Uinf)
			Lubrication_RSU_kernel<<< grid, threads >>>(
								d_Stress,
								d_rhs_temp,
								d_pos,
								res_data->body_tag,
								res_data->local_index,
								d_group_members,
								group_size, 
								box,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist, 
								res_data->table_dist,
								res_data->table_vals,
								res_data->table_min,
								res_data->table_dr,
								res_data->rlub
								);
			// + RSE_nf : Einf
			Lubrication_RSE_kernel<<< grid, threads >>>(
								d_Stress,
								shear_rate,
								res_data->body_tag,
								group_size, 
								d_group_members,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist, 
								d_pos,
								box,
								res_data->table_dist,
								res_data->table_vals,
								res_data->table_min,
								res_data->table_dr
								);
      		} //check for lubrication stress calculation
			
                //Compute constrained stresslet
		AddBeadStress_kernel<<< grid, threads >>>(
							&d_Stress[5*group_size],
							d_force,
							res_data->rel_pos,
							group_size,
							d_group_members
							);

    	} //check for stress calculation

  // Clean up
  d_rhs_temp = NULL;
  d_EinfForce = NULL;
  d_rhs = NULL;
  d_solution = NULL;
  d_force = NULL;
}
