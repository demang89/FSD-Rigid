// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore
// Zhouyang Ge

#include "Brownian_NearField.cuh"
#include "Precondition.cuh"
#include "Lubrication.cuh"

#include "Helper_Brownian.cuh"
#include "Helper_Debug.cuh"
#include "Helper_Precondition.cuh"

#include <hoomd/RNGIdentifiers.h>
#include <hoomd/RandomNumbers.h>
using namespace hoomd;

#include <stdio.h>
#include <math.h>

#include <curand.h>
#include <cuda_runtime.h>

// LAPACK and CBLAS
#include "lapacke.h"
#include "cublas_wrappers.hpp"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! 
	\file Brownian_NearField.cu
	\brief Defines functions to compute the near-field Brownian Forces
*/

/*!
  	Generate random numbers on particles for Near-field calculation
	
	d_Psi_nf	(output) uniform random vector
        group_size	(input)  number of particles
	seed		(input)  seed for random number generation
	T		(input)  Temperature
	dt		(input)  Time step
*/
__global__ void Brownian_NearField_RNG_kernel(
						Scalar *d_Psi_nf,
						unsigned int group_size,
						const uint64_t timestep,
						const unsigned int seed
						//const float T,
						//const float dt
						){

	// Thread index
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// Check if thread is in bounds, and if so do work
	if (idx < group_size) {
		hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::TwoStepBD, timestep, seed), hoomd::Counter(idx));
		hoomd::NormalDistribution<Scalar> uniform(1.0);
    
		// Generate random numbers and assign to global output
		d_Psi_nf[ 6 * idx     ] = uniform(rng);
		d_Psi_nf[ 6 * idx + 1 ] = uniform(rng);
		d_Psi_nf[ 6 * idx + 2 ] = uniform(rng);
		d_Psi_nf[ 6 * idx + 3 ] = uniform(rng);
		d_Psi_nf[ 6 * idx + 4 ] = uniform(rng);
		d_Psi_nf[ 6 * idx + 5 ] = uniform(rng);

	} // Check if thread is in bounds

}


/*!
	Use Lanczos method to compute RFU^0.5 * psi

	This method is detailed in the publication:
	Edmond Chow and Yousef Saad, PRECONDITIONED KRYLOV SUBSPACE METHODS FOR
	SAMPLING MULTIVARIATE GAUSSIAN DISTRIBUTIONS, SIAM J. Sci. Comput., 2014

	d_FBnf			(output) near-field Brownian force
	d_psi			(input)  uniform random vector
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	dt			(input)  integration timestep
	pBuffer			(input)  scratch buffer space for preconditioner
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	res_data		(input)  structure containing lubrication calculation information
	work_data		(input)  structure containing workspaces

*/

void Brownian_NearField_Chow_Saad( Scalar *d_y,  // output: near-field Brownian force
				   Scalar *d_x,  // input: random Gaussian variables
				   Scalar4 *d_pos,
				   unsigned int *d_group_members,
				   unsigned int group_size,
				   const BoxDim& box,
				   Scalar dt,
				   void *pBuffer,
				   KernelData *ker_data,
				   BrownianData *bro_data,
				   ResistanceData *res_data,
				   WorkData *work_data)
{
  // cuBLAS handle
  cublasHandle_t blasHandle = work_data->blasHandle;

  // Constants
  int stride = 6;
  int numel = 6 * group_size;      //size of v1,v2,...,vm_max, d_x, d_y
  int m = bro_data->m_Lanczos_nf - 1;  //number of Lanczos iterations in step 1 (either same as last time or reset in Stokes.cc)
  m = m < 1 ? 1 : m;
  int m_max = 100;                 //m_max-1 is the maximum size of Tm at the end of step 2 (set to 100 in Stokes.cc)

  //debug
  if ( m >= m_max-1 )
    {
      printf("Illegal condition: m >= m_max-1. Program aborted.");
      exit(1);
    }
  
  // Host vectors for the main and sub-diagonal values of Tm
  Scalar *h_alpha  = (Scalar *)malloc( (m_max)*sizeof(Scalar) );
  Scalar *h_beta   = (Scalar *)malloc( (m_max+1)*sizeof(Scalar) );
  Scalar *h_alpha1 = (Scalar *)malloc( (m_max)*sizeof(Scalar) );  //buffer
  Scalar *h_beta1  = (Scalar *)malloc( (m_max+1)*sizeof(Scalar) );  //buffer

  // Set the first element of beta to 0
  h_beta[0] = 0.0;

  // Set the tolerance for beta (less than 1e-6 even for single precision because ||vm|| can be << 1)
  Scalar tol_beta = 1e-10; 

  // Buffer vector for checking convergence
  Scalar *d_y0 = work_data->bro_nf_FB_old;  

  // Lanczos basis vectors V = [v0, v1, v2, ..., vm_max], v0 is a placeholder
  Scalar *d_v = work_data->bro_nf_v;
  Scalar *d_V = work_data->bro_nf_V;
	
  // Zero out v0
  cudaMemset( d_V, 0, numel*sizeof(Scalar));

  // Initialize v1 = d_x / ||d_x||
  Scalar xnorm;
  cublas::nrm2<Scalar>( blasHandle, numel, d_x, 1, &xnorm );
  cudaMemcpy( &d_V[numel], d_x, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );
  
  Scalar scale = 1.0 / xnorm;
  cublas::scal<Scalar>( blasHandle, numel, &scale, &d_V[numel], 1 ); 
  
  //
  // Step 1: Build Vm and Tm via the Lanczos process
  //
  for ( int j = 0; j < m; ++j )  //iterate at most m times 
    {
      // d_v = A * d_V[j+1]
      Precondition_Brownian_RFUmultiply( d_v,  // output
                                         &d_V[ (j+1)*numel ],   // input
                                         d_pos,
                                         d_group_members,
                                         group_size,
                                         box,
                                         pBuffer,
                                         ker_data,
                                         res_data );

      Scalar scale = -1.0 * h_beta[j];
      cublas::axpy<Scalar>( blasHandle, numel, &scale, &d_V[ j*numel ], 1, d_v, 1 ); // d_v -= beta * d_V[j]
      cublas::dot<Scalar>( blasHandle, numel, &d_V[ (j+1)*numel ], 1, d_v, 1, &h_alpha[j]); // alpha = d_V[j+1] \cdot d_v

      scale = -1.0 * h_alpha[j];
      cublas::axpy<Scalar>( blasHandle, numel, &scale, &d_V[ (j+1)*numel ], 1, d_v, 1 ); // d_v -= alpha * d_V[j+1)
      cublas::nrm2<Scalar>( blasHandle, numel, d_v, 1, &h_beta[j+1] ); // beta = || d_v ||)

      // Stop if beta becomes very small
      if ( h_beta[j+1] < tol_beta )
      {
        m = j;  //plus 1 because one iteration was done when j=0
        //printf("for y0, it stopped earlier at %d\n",j);
        break;
      }

      scale = 1.0 / h_beta[j+1];
      cublas::scal<Scalar>( blasHandle, numel, &scale, d_v, 1 );  //d_v /= beta
      cudaMemcpy( &d_V[(j+2)*numel], d_v, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice ); // Store current basis vector
    }  

  // Step 2 : compute d_y0
  Sqrt_multiply( &d_V[ numel ],  //input
                  h_alpha,        //input
                  &h_beta[1],	 //input
                  h_alpha1,	 //input (buffer)
                  h_beta1,        //input (buffer)
                  m,              //input 
                  d_y,           //output
                  stride,
                  group_size,
                  ker_data,
                  work_data 
                  );

  cudaMemcpy( d_y0, d_y, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

  // Step 3 : Keep adding to basis until convergence
  Scalar error = 1.0;
  Scalar ynorm = 1.0;
  cublas::nrm2<Scalar>( blasHandle, numel, d_y0, 1, &ynorm );

  while( error > bro_data->tol and m < m_max-1 )
    {
      // d_v = A * d_V[j]
      Precondition_Brownian_RFUmultiply( d_v,  // output
                                         &d_V[(m+1)*numel ],   // input
                                         d_pos,
                                         d_group_members,
                                         group_size,
                                         box,
                                         pBuffer,
                                         ker_data,
                                         res_data );

      Scalar scale = -1.0 * h_beta[m];
      cublas::axpy<Scalar>( blasHandle, numel, &scale, &d_V[ m*numel ], 1, d_v, 1 ); // d_v -= beta * d_V[m]
      cublas::dot<Scalar>( blasHandle, numel, &d_V[ (m+1)*numel ], 1, d_v, 1, &h_alpha[m]); // alpha = d_V[m+1] \cdot d_v

      scale = -1.0 * h_alpha[m];
      cublas::axpy<Scalar>( blasHandle, numel, &scale, &d_V[ (m+1)*numel ], 1, d_v, 1 );  // d_v -= alpha * d_V[m+1)
      cublas::nrm2<Scalar>( blasHandle, numel, d_v, 1, &h_beta[m+1] ); // beta = || d_v ||)

      // Stop if beta becomes very small
      if ( h_beta[m+1] < tol_beta )
        {
          break;
        }

      scale = 1.0 / h_beta[m+1];
      cublas::scal<Scalar>( blasHandle, numel, &scale, d_v, 1 ); //d_v /= beta

      // Store current basis vector
      cudaMemcpy( &d_V[(m+2)*numel], d_v, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

      // Compute the new approximate solution, d_y
      Sqrt_multiply( &d_V[ numel ],  //input
                      h_alpha,        //input
                      &h_beta[1],	     //input
                      h_alpha1,	     //input (buffer)
                      h_beta1,        //input (buffer)
                      m+1,            //input 
                      d_y,            //output
                      stride,
                      group_size,
                      ker_data,
                      work_data 
                      );
      	
      // Compute relative error = || d_y0 - d_y || / || d_y ||
      scale = -1.0;
      cublas::axpy<Scalar>( blasHandle, numel, &scale, d_y, 1, d_y0, 1 );  //d_y0 is modified in place
      cublas::nrm2<Scalar>( blasHandle, numel, d_y0, 1, &error );
      cublas::nrm2<Scalar>( blasHandle, numel, d_y,  1, &ynorm );
      if (ynorm == 0.0) ynorm = 1.0;
      error /= ynorm;

      // Update solution
      cudaMemcpy( d_y0, d_y, numel*sizeof(Scalar), cudaMemcpyDeviceToDevice );

      // Increment m
      ++m;
	
    }

  //printf("stopped at %d while m_in is %d\n",m,m_in);
  // Save the number of required iterations (minus 1 because incremented at the end)
  bro_data->m_Lanczos_nf = m;

  //// Undo the preconditioning so that the result has the proper variance
  /*Precondition_Brownian_Undo( d_y,       //input/output
  			      group_size,
			      pBuffer,
  			      ker_data,
  			      res_data );*/

  // Rescale by original norm of d_x
  scale = xnorm * sqrt( 2.0 * bro_data->T); //Deepak:removed dt from here for better saddle accuracy
  cublas::scal<Scalar>( blasHandle, numel, &scale, d_y, 1 );

  // Clean up
  free(h_alpha);
  free(h_alpha1);
  free(h_beta);
  free(h_beta1);
		
}



/*
	Wrap all the functions required to compute the near-field Brownian force.
	
	d_FBnf			(output) near-field Brownian force
	d_pos			(input)  particle positions
	d_group_members		(input)  ID of particle within integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	dt			(input)  integration timestep
	ker_data		(input)  structure containing kernel launch information
	bro_data		(input)  structure containing Brownian calculation information
	res_data		(input)  structure containing lubrication calculation information
	work_data		(input)	 structure containing workspaces
*/
void Brownian_NearField_Force(Scalar *d_FBnf, // output
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
			      )
{


  // Initialize vectors
  Scalar *d_Psi_nf = work_data->bro_nf_psi;

  //// Generate the random vectors on each particle
  //// Kernel Information
  dim3 grid = ker_data->particle_grid;
  dim3 threads = ker_data->particle_threads;
  Brownian_NearField_RNG_kernel<<<grid,threads>>>( 
                                                  d_Psi_nf,  //output
                                                  group_size,
                                                  timestep,
                                                  bro_data->seed_nf
                                                  );
  
  // Apply the Chow & Saad method to sample the near-field force
  Brownian_NearField_Chow_Saad( d_FBnf,   //output
			        d_Psi_nf, //input
			        d_pos,
			        d_group_members,
			        group_size,
			        box,
			        dt,
			        pBuffer,
			        ker_data,
			        bro_data,
			        res_data,
			        work_data);
		
  // Clean Up
  d_Psi_nf = NULL;
}
