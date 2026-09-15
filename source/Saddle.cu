// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore


#include "Saddle.cuh"
#include "Lubrication.cuh"
#include "Precondition.cuh"
#include "Mobility.cuh"
#include "Solvers.cuh"
#include "Wrappers.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Mobility.cuh"
#include "Helper_Precondition.cuh"
#include "Helper_Saddle.cuh"

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>

#include "hoomd/TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <stdlib.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! command to convert floats or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

/*
	Define the saddle point matrix describing Stokesian Dynamics,
	i.e. it describes the relationship Ax=b (rather than constructing the matrix A).
*/

/*! 
	Matrix-vector operation associated with the saddle point matrix solve

	d_b			(output) output of matrix-vector product (a vector of size 17N)
	d_x			(input)  input of matrix-vector product
	d_pos			(input)  positions of the particles, actually they are fetched on texture memory
	d_group_members		(input)  index array to global HOOMD tag on each particle
	group_size		(input)  size of the group, i.e. number of particles
	box			(input)  array containing box dimensions
	ker_data		(input)  structure containing information for kernel launches
	mob_data		(input)  structure containing information for mobility calculations
	res_data		(input)  structure containing information for resistance calculation

*/

//zhoge// Referenced by cuspSaddle in Wrappers.cuh
void Saddle_Multiply( 
			Scalar *d_b, // output
			Scalar *d_x, // input
			Scalar4 *d_pos,
			unsigned int *d_group_members,
			unsigned int group_size,
			const BoxDim& box,
			KernelData *ker_data,
			MobilityData *mob_data,
			ResistanceData *res_data,
			WorkData *work_data
			){
	
	// Kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;
	
	unsigned int array_size = 11*group_size + 6*res_data->N_rigid;
	// Set output to zero to start (size 17N)
	cudaMemset(d_b, 0, array_size * sizeof(Scalar));

	//Deepak:expand aggregate velocities
	Scalar *d_x_temp = res_data->Scratch8;
	cudaMemset(d_x_temp, 0, 6 * group_size * sizeof(Scalar));
	sigma_transpose_kernel<<< grid, threads >>>(&d_x[11*group_size], d_x_temp,
						res_data->rel_pos, res_data->body_tag, d_group_members, group_size);
	
	// Do the mobility multiplication, M^ff * F => d_b[0:11N]
	Mobility_GeneralizedMobility(
					d_b, //output (temporary, modified next)
					d_x, //input (generalized forces)
					d_pos,
					res_data->local_index,
					d_group_members,
					group_size,
					box,
					ker_data,
					mob_data,
					work_data
					);
	
	// M^ff*F + B*U => RHS[0:11N]. Effectively, d_b[0:6N] += d_x[11N:17N]
	Saddle_AddFloat_kernel<<<grid,threads>>>(d_b,
						 d_x_temp,
						 d_b,                 //output
						 1.0, 1.0,
						 group_size, 6 );

	
	Scalar *d_b_temp = res_data->Scratch6;
	cudaMemset(d_b_temp, 0, 6 * group_size * sizeof(Scalar));

	// Do the resistance multiplication, R_FU^nf * U => d_b[11N:17N]
	if(res_data->has_lubrication){
	     Lubrication_RFU_kernel<<<grid,threads>>>(
							d_b_temp, // output (temporary, modified next)
							d_x_temp, // input (relative velocity)
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
 
	
	// B^T*F - R_FU*U => RHS[11N:17N]. Effectively, d_b[11N:17N] = d_x[0:6N] - d_b[11N:17N]
	Saddle_AddFloat_kernel<<<grid,threads>>>(
						d_x,
						d_b_temp,
						d_b_temp,  //output
						1.0, -1.0,
						group_size, 
						6 
						);

	sigma_kernel<<< grid, threads >>>(d_b_temp, &d_b[11*group_size],
                        res_data->rel_pos, res_data->body_tag, d_group_members, group_size);

	d_x_temp = nullptr;
	d_b_temp = nullptr;
}



/*!
	Matrix-vector operation for saddle point preconditioner
		x = P \ b

	(zhoge: P \ b means P^-1 * b)

	!!! In order for this to work with cusp, the operator must be
	    able to do the linear transformation in place! (gmres.inl line 143 in CUSP)
	
	d_x			(output) Solution of preconditioner
	d_b			(input)  RHS of preconditioner solve
	group_size		(input)  size of the group, i.e. number of particles
	ker_data		(input)  structure containing information for kernel launches
	res_data		(input)  structure containing information for resistance calculation

*/
void Saddle_Preconditioner(	
				Scalar *d_x, 		// output
				Scalar *d_b, 		// input
				unsigned int *d_group_members,
				int group_size,
				void *pBuffer,
				KernelData *ker_data,
				ResistanceData *res_data
				){

	// Get kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Get pointer to scratch array
	Scalar *d_Scratch2 = res_data->Scratch2;

	//Deepak : added for rigid body simulation
	int group_size_b = res_data->N_rigid;
	Scalar *d_x_temp = res_data->Scratch1;
	cudaMemset(d_x_temp, 0, 6 * group_size_b * sizeof(Scalar));

	sigma_kernel<<< grid, threads >>>(d_b, d_x_temp,
					res_data->rel_pos, res_data->body_tag, d_group_members, group_size);

	Saddle_AddFloat_kernel<<<grid,threads>>>( d_x_temp, &d_b[11*group_size], d_x_temp,
                                                1.0, -1.0, group_size_b, 6 );

	Scalar *d_Scratch6 = res_data->Scratch7;
        Precondition_Saddle_RFUmultiply(
                                &d_Scratch2[11*group_size], // output
                                d_x_temp,                 // input
                                d_Scratch6,        // intermediate storage
                                res_data->prcm,
                                group_size_b,
                                res_data->nnz,
                                res_data->L_RowPtr,
                                res_data->L_ColInd,
                                res_data->L_Val,
                                res_data->spHandle,
                                res_data->spStatus,
                                res_data->descr_L,
                                res_data->info_L,
                                res_data->info_Lt,
                                res_data->trans_L,
                                res_data->trans_Lt,
                                res_data->policy_L,
                                res_data->policy_Lt,
                                pBuffer,
                                ker_data->rigid_grid,
                                ker_data->rigid_threads
                                );

	sigma_transpose_kernel<<< grid, threads >>>(&d_Scratch2[11*group_size], d_Scratch2,
						res_data->rel_pos, res_data->body_tag, d_group_members, group_size);

	Saddle_AddFloat_kernel<<<grid,threads>>>( d_b, d_Scratch2, d_Scratch2, 1.0, -1.0, group_size, 6 );
	cudaMemcpy(&d_Scratch2[6*group_size], &d_b[6*group_size], 5*group_size*sizeof(Scalar), cudaMemcpyDeviceToDevice );

	cudaMemcpy(d_x, d_Scratch2, (11*group_size+6*group_size_b)*sizeof(Scalar), cudaMemcpyDeviceToDevice );

    	d_x_temp = NULL;
    	d_Scratch2 = NULL;
	d_Scratch6 = NULL;
}
