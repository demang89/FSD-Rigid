// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Wrappers.cuh"
#include "Solvers.cuh"
#include "Lubrication.cuh"

#include <cusp/hyb_matrix.h>
#include <cusp/monitor.h>
#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/linear_operator.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/gmres.h>
#include <cusp/krylov/cg.h>
#include <cusp/multiply.h>

#include <cusp/blas.h>

#include <cusparse.h>
#include <cusolverSp.h>

#include <stdio.h>

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
	Construct the saddle point matrix, the preconditioner matrix, and do the
	preconditioned solve, all in one wrapper function.

	d_rhs			(input)  right-hand side for the saddle point solve
	d_solution		(output) solution to the saddle point solve
	d_pos			(input)  particle positions
	d_group_members		(input)  indices of particles in the integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	pBuffer			(input)  buffer space for cuSPARSE operations in preconditioner
	ker_data		(input)  structure containing information for kernel launches
	mob_data		(input)  structure containing information for mobility calculation
	res_data		(input)  structure containing information for resistance calculation

*/
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
			){

	unsigned int array_size = 11*group_size + 6*res_data->N_rigid;

	// Set up CUSP saddle point matrix object
	cuspSaddle SADDLE(
				d_pos,
				d_group_members,
				group_size,
				array_size,
				box,
				ker_data,
				mob_data,
				res_data,
				work_data
				);

        // Set up CUSP preconditioner matrix object
	cuspSaddlePreconditioner PRECONDITIONER(
						d_group_members,
						group_size,
						array_size,
						pBuffer,
						ker_data,
						res_data
						);

	// Wrap raw pointers for solution (initial guess) and RHS with thrust::device_ptr
	thrust::device_ptr<Scalar> d_x( d_solution );
	thrust::device_ptr<Scalar> d_b( d_rhs );

	// Wrap thrust device pointers in cusp array1d_view
	typedef typename cusp::array1d_view< thrust::device_ptr<Scalar> > DeviceArrayView;
	DeviceArrayView x (d_x, d_x + array_size );
	DeviceArrayView b (d_b, d_b + array_size );

	// CUSP Solver Monitor
	// 	rhs vector	= b
	//	tol      	= 1E-3  (specified in the run.py /zhoge)
	//
	//      Converge if residual norm || b - A*x || <= abs_tol + rel_tol * || b ||
	//
	int iter_limit = 1000;
	Scalar rel_tol  = tol;
	Scalar abs_tol  = 0;
	bool verbose_flag = false;
	cusp::monitor<Scalar> monitor(b, iter_limit, rel_tol, abs_tol, verbose_flag);

	// solve the linear system A * x = b using GMRES
	//
	// Smaller values of the restart parameter reduce memory requirements but 
	// also worsen the convergence. 
	int restart = 200;
	cusp::krylov::gmres( SADDLE, x, b, restart, monitor, PRECONDITIONER);
	//cusp::krylov::cg( SADDLE, x, b, monitor, PRECONDITIONER);
	
	std::cout << "Iteration Count: " << monitor.iteration_count() << ", Residual: " << monitor.residual_norm() << std::endl; 
	if (!monitor.converged())
	  {
	    printf ("  GMRES solver failed to converge. Iterations = %5lu, residual = %10.3e \n",
		    monitor.iteration_count(), monitor.residual_norm() );
	  }
	
}
