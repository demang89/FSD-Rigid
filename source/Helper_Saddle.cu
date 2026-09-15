// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore


#include "Helper_Saddle.cuh"

#include <cusparse.h>

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


/*! \file Helper_Saddle.cu
	Helper functions to perform the additions and operations required in the saddle point
	matrix calculations
*/

/*! 
	Zero the output for the saddle point multiplication
	
	d_b	(input/output) 	vector zeroed upon output
   	N	(input) 	number of particles
*/
__global__ void Saddle_ZeroOutput_kernel( 
					Scalar *d_b,
					unsigned int N
					){

	// Thread index
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if thread is inbounds
	if ( tid < N ) {
	
		// Do the zeroing
		for ( int ii = 0; ii < 17; ii++ ){ 
			d_b[ 17*tid + ii ] = 0.0f;
	      	}  
	
	}
}


/*!
        Direct addition of two float arrays

        C = a*A + b*B
        C can be A or B, so that A or B will be overwritten

        d_a		(input)  input vector, A
        d_b		(input)  input vector, B
        d_c		(output) output vector, C
        coeff_a		(input)  scaling factor for A, a
        coeff_b		(input)  scaling factor for B, b
        N		(input)  length of vectors
	stride		(input)  number of repeats
*/
__global__ void Saddle_AddFloat_kernel( 
					Scalar *d_a, 
					Scalar *d_b,
					Scalar *d_c,
					Scalar coeff_a,
					Scalar coeff_b,
					unsigned int N,
					int stride
					){

	// Thread index
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (idx < N) {
       
		for ( int ii = 0; ii < stride; ++ii ){
			
			// Index for current striding
			int ind = stride * idx + ii;

			// Do addition
			d_c[ ind ] = coeff_a * d_a[ ind ] + coeff_b * d_b[ ind ];
 
		}

        }
}

/*!
        Split generalized force into force/torque/stresslet

	d_generalF	(input)  11N vector of generalized force (force/torque first 6N, stresslet last 5N)
	d_net_force	(output) linear force
	d_TorqueStress	(output) torque and stresslet
	N		(input)  number of particles

*/
__global__ void Saddle_SplitGeneralizedF_kernel( 	
						Scalar *d_generalF, 
						Scalar4 *d_net_force,
						Scalar4 *d_TorqueStress,
						unsigned int group_size
						){
	// Thread index
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (tidx < group_size) {

		int ind1 = 6*tidx;
		int ind2 = 6*group_size + 5*tidx; 
 
		// 
		Scalar f1 = d_generalF[ ind1 + 0 ];
		Scalar f2 = d_generalF[ ind1 + 1 ];
		Scalar f3 = d_generalF[ ind1 + 2 ];
		Scalar l1 = d_generalF[ ind1 + 3 ];
		Scalar l2 = d_generalF[ ind1 + 4 ];
		Scalar l3 = d_generalF[ ind1 + 5 ];
		Scalar s1 = d_generalF[ ind2 + 0 ];
		Scalar s2 = d_generalF[ ind2 + 1 ];
		Scalar s3 = d_generalF[ ind2 + 2 ];
		Scalar s4 = d_generalF[ ind2 + 3 ];  //zhoge: Syz
		Scalar s5 = d_generalF[ ind2 + 4 ];  //zhoge: Syy

		d_net_force[ tidx ] = make_scalar4( f1, f2, f3, 0.0 );
		d_TorqueStress[ 2*tidx + 0 ] = make_scalar4( l1, l2, l3, s1 );
		d_TorqueStress[ 2*tidx + 1 ] = make_scalar4( s2, s3, s4, s5 );

        }
}

/*!
        Combine velocity/angular velocity/rate of strain into generalized velocity

	d_generalU	(output) 11N vector of generalized velocity (first 6N) and trate of strain (last 5N)
	d_vel		(input)  linear velocity
	d_AngvelStrain	(input)  angular velocity and rate of strain
	N		(input)  number of particles

*/
__global__ void Saddle_MakeGeneralizedU_kernel( 	
						Scalar *d_generalU, 
						Scalar4 *d_vel,
						Scalar4 *d_AngvelStrain,
						unsigned int group_size
						){
	// Thread index
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        
	// Check if thread is in bounds
	if (tidx < group_size) 
	{
		Scalar4 vel = d_vel[ tidx ];
		Scalar4 AS1 = d_AngvelStrain[ 2*tidx + 0 ];
		Scalar4 AS2 = d_AngvelStrain[ 2*tidx + 1 ];

		int ind1 = 6*tidx;
		int ind2 = 6*group_size + 5*tidx;      
 
		d_generalU[ ind1 + 0 ] = vel.x;   // U_x - U^infty
		d_generalU[ ind1 + 1 ] = vel.y;   // U_y - U^infty
		d_generalU[ ind1 + 2 ] = vel.z;   // U_z - U^infty
		d_generalU[ ind1 + 3 ] = AS1.x;   // Omega_x - Omega^infty
		d_generalU[ ind1 + 4 ] = AS1.y;   // Omega_y - Omega^infty
		d_generalU[ ind1 + 5 ] = AS1.z;   // Omega_z - Omega^infty
		d_generalU[ ind2 + 0 ] = AS1.w;   // E_xx - E_zz
		d_generalU[ ind2 + 1 ] = AS2.x;   // E_xy * 2
		d_generalU[ ind2 + 2 ] = AS2.y;   // E_xz * 2
		d_generalU[ ind2 + 3 ] = AS2.z;   // E_yz * 2
		d_generalU[ ind2 + 4 ] = AS2.w;   // E_yy - E_zz

    }
}

__global__ void sigma_kernel(
				Scalar *d_a,
                             	Scalar *d_b,
                             	Scalar3 *d_c,
                             	int *d_body_tag,
                             	unsigned int *d_group_members,
                             	unsigned int group_size
				){
    // Thread index
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < group_size) {
            int idx = d_group_members[tidx];
        	int bidx = d_body_tag[idx];
			Scalar3 c = make_scalar3(d_c[idx].x, d_c[idx].y, d_c[idx].z);

			// Read d_a values
			Scalar a0 = d_a[6 * tidx + 0];
			Scalar a1 = d_a[6 * tidx + 1];
			Scalar a2 = d_a[6 * tidx + 2];
			Scalar a3 = d_a[6 * tidx + 3];
			Scalar a4 = d_a[6 * tidx + 4];
			Scalar a5 = d_a[6 * tidx + 5];

			// Compute the torque-corrected values
			Scalar t3 = a3 - c.z * a1 + c.y * a2;
			Scalar t4 = a4 + c.z * a0 - c.x * a2;
			Scalar t5 = a5 - c.y * a0 + c.x * a1;

			// Use atomic adds to safely accumulate into d_b
			atomicAdd(&d_b[6 * bidx + 0], a0);
			atomicAdd(&d_b[6 * bidx + 1], a1);
			atomicAdd(&d_b[6 * bidx + 2], a2);
			atomicAdd(&d_b[6 * bidx + 3], t3);
			atomicAdd(&d_b[6 * bidx + 4], t4);
			atomicAdd(&d_b[6 * bidx + 5], t5);
    }
}


__global__ void sigma_transpose_kernel( 
					Scalar *d_a,
                                        Scalar *d_b,
                                        Scalar3 *d_c,
                                        int *d_body_tag,
                                        unsigned int *d_group_members,
                                        unsigned int group_size
                                        ){

        // Thread index
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

        // Check if thread is in bounds
        if (tidx < group_size) {
                        int idx = d_group_members[tidx];
                        int bidx = d_body_tag[idx];
                        Scalar3 c = make_scalar3(d_c[idx].x, d_c[idx].y, d_c[idx].z);
                        d_b[6*tidx    ] = d_a[6 * bidx    ] + c.z * d_a[6 * bidx + 4] - c.y * d_a[6 * bidx + 5] ;
                        d_b[6*tidx + 1] = d_a[6 * bidx + 1] - c.z * d_a[6 * bidx + 3] + c.x * d_a[6 * bidx + 5];
                        d_b[6*tidx + 2] = d_a[6 * bidx + 2] + c.y * d_a[6 * bidx + 3] - c.x * d_a[6 * bidx + 4];
                        d_b[6*tidx + 3] = d_a[6 * bidx + 3];
                        d_b[6*tidx + 4] = d_a[6 * bidx + 4];
                        d_b[6*tidx + 5] = d_a[6 * bidx + 5];
        }

}


__global__ void subtract_uinf_kernel( 
					Scalar *d_a,
					Scalar4 *d_pos,
					unsigned int *d_group_members,
					unsigned int group_size,
					Scalar shear_rate
                    			){

    // Thread index
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is in bounds
    if (tidx < group_size) {
              int idx = d_group_members[tidx];
              d_a[6*tidx    ] -= (shear_rate * d_pos[idx].y);
              d_a[6*tidx + 5] += (0.5 * shear_rate);
    }

}
