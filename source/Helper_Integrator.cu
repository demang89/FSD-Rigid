// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Integrator.cuh"

#include <hoomd/RNGIdentifiers.h>
#include <hoomd/RandomNumbers.h>
#include "hoomd/TextureTools.h"
using namespace hoomd;

#include <stdio.h>
#include <math.h>

#include "lapacke.h"
#include "cblas.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif


/*! 
	Helper_Integrator.cu

	Helper functions for saddle point integration
*/
	
/*!
  	Generate random numbers on particles.
	
	d_psi		(output) random vector
        n		(input)  number of particles
	timestep	(input)  length of time step
	seed		(input)  seed for random number generation

*/
__global__ void Integrator_RFD_RandDisp_kernel(
						Scalar *d_psi,
						unsigned int N,
						int stride,
						const uint64_t timestep,
						const unsigned int seed
						){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds
	if (idx < N) {
		//Deepak:included new randomgenerator class
		// Initialize random seed
		//Scalar sqrt3 = 1.732050807568877;
		RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepBD, timestep, seed),hoomd::Counter(idx));
		NormalDistribution<Scalar> uniform(1.0);
		// Write to output
		for (int i=0;i<stride;i++){
			d_psi[ stride*idx + i ] = uniform(rng);
		}
	}

}

/*! 
	The output velocity

	d_b	(output) output vector
   	N 	(input)  number of particles

*/
__global__ void Integrator_ZeroVelocity_kernel( 
						Scalar *d_b,
						unsigned int N
						){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < N ) {
	
		d_b[ 6*tid + 0 ] = 0.0;
		d_b[ 6*tid + 1 ] = 0.0;
		d_b[ 6*tid + 2 ] = 0.0;
		d_b[ 6*tid + 3 ] = 0.0;
		d_b[ 6*tid + 4 ] = 0.0;
		d_b[ 6*tid + 5 ] = 0.0;
	
	}
}

/*! 
	Add rate of strain from shearing to the right-hand side of the saddle point solve
	d_b		(input/output) 	right-hand side vector
	shear_rate 	(input) 	shear rate of applied deformation
	B2              (input)         coefficient of B2 mode (spherical squirmers)
	d_ori           (input)         particle orientation (unit vector)
   	N 		(input)  	number of particles

*/
//Deepak:modified to include uinf on rhs
__global__ void Integrator_AddStrainRate_kernel( 
						Scalar *d_b,
						Scalar shear_rate,
						Scalar4 *d_pos,
						unsigned int *d_group_members,
						unsigned int group_size,
						Scalar3 *d_rel_pos
						){

	// Thread index
	unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tidx < group_size ) {

		// Particle ID
		unsigned int idx = d_group_members[tidx];
		d_b[ 6 * tidx + 0 ] += (0.5 * shear_rate * d_rel_pos[idx].y);
		d_b[ 6 * tidx + 1 ] += (0.5 * shear_rate * d_rel_pos[idx].x);

		// Index into array
		unsigned int ind = 6*group_size + 5*tidx;
	        
		// Add ambient strain rate Einf (E_xy = E_yx = shear_rate/2, all else 0)
		d_b[ ind + 0 ] += 0.0;        // E_xx - E_zz
		d_b[ ind + 1 ] += shear_rate; // E_xy * 2   
		d_b[ ind + 2 ] += 0.0;	      // E_xz * 2   
		d_b[ ind + 3 ] += 0.0;	      // E_yz * 2   
		d_b[ ind + 4 ] += 0.0;	      // E_yy - E_zz
		
	}
}
