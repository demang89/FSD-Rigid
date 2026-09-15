// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore

#include "Helper_Stokes.cuh"  //zhoge: This includes HOOMDMath.h, which includes cmath

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


/*! \file Helper_Stokes.cu
    	\brief Helper functions required for data handling in Stokes.cu
*/
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
		){

        // Thread idx
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;

        // Do work if thread is in bounds
        if (tidx < group_size) {
        	unsigned int j = d_group_members[tidx];
        	Scalar4 posi = d_pos[j];
        	unsigned int central_tag = d_body_data[j];
        	unsigned int central_idx = d_rtag[central_tag];

        	Scalar4 posj = d_pos[central_idx];
        	Scalar3 dx = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
        	dx = box.minImage(dx);
        	d_rel_pos[j] = dx;
        	d_body_tag[j] = central_tag;
        	d_local_index[j] = tidx;

                vec3<Scalar> local_pos(dx);
                quat<Scalar> q(d_ori[central_idx]);
                vec3<Scalar>  dr = rotate(conj(q),local_pos);
                d_rel_pos_bf[j] = vec_to_scalar3(dr);
		//printf("(%d,%d),",j,central_tag);
	}
}


__global__ void Stokes_update_rel_pos(
                Scalar4 *d_pos,
                const unsigned int* d_rtag,
                int *d_body_tag, //output
                Scalar3 *d_rel_pos,  //output
                unsigned int group_size,
                unsigned int *d_group_members,
                const BoxDim box
                ){

        // Thread idx
        int tidx = blockDim.x * blockIdx.x + threadIdx.x;

        // Do work if thread is in bounds
        if (tidx < group_size) {
                unsigned int j = d_group_members[tidx];
                Scalar4 posi = d_pos[j];
                unsigned int central_tag = d_body_tag[j];
                unsigned int central_idx = d_rtag[central_tag];

                Scalar4 posj = d_pos[central_idx];
                Scalar3 dx = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
                dx = box.minImage(dx);
                d_rel_pos[j] = dx;
        }
}


/*!
	Initialize the total applied force and torque using the net_force
	vector from HOOMD which contains the contributions from external
	and interparticle potentials

	d_net_force		(input)  HOOMD force vector
	d_AppliedForce		(output) Total force experience by the particles
	group_size		(input)  length of vectors
	d_group_members		(input)  index into vectors

*/
__global__ void Stokes_SetForce_kernel(
					Scalar4 *d_net_force,
					Scalar4 *d_net_torque,
					Scalar   *d_AppliedForce,
					unsigned int group_size,
					unsigned int *d_group_members
					){

	// Thread idx
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Do work if thread is in bounds
	if (tidx < group_size) {

		unsigned int idx = d_group_members[ tidx ];
		
		Scalar4 net_force = d_net_force[ idx ];
		Scalar4 net_torque = d_net_torque[ idx ];

		d_AppliedForce[ 6*tidx     ] = net_force.x;
		d_AppliedForce[ 6*tidx + 1 ] = net_force.y;
		d_AppliedForce[ 6*tidx + 2 ] = net_force.z;
		d_AppliedForce[ 6*tidx + 3 ] = net_torque.x;
		d_AppliedForce[ 6*tidx + 4 ] = net_torque.y;
		d_AppliedForce[ 6*tidx + 5 ] = net_torque.z;

	}
}

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
						){

  // Thread idx
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  // Do work if thread is in bounds
  if (tidx < group_size) {
    
    unsigned int idx = d_group_members[ tidx ];

    Scalar4 posi = d_pos[idx];  // position

    // Interparticle force parameters
    Scalar h_rough = sqrt(epsq);       //roughness height

    //Deepak: for DLVO
    Scalar radsubsq = 0.0;//(a1^2-a2^2)
    Scalar radsumsq = 2.0;//(a1^2+a2^2)

    // Interparticle force
    Scalar F_x = F_ext * 0.0;
    Scalar F_y = F_ext * 0.0;
    Scalar F_z = F_ext * 1.0;

    // (along z direction)
    Scalar T_x = T_ext * 0.0;
    Scalar T_y = T_ext * 0.0;
    Scalar T_z = T_ext * 1.0;

     // Neighborlist arrays
    unsigned int head_idx = d_headlist[ idx ]; // Location in head array for neighbors of current particle
    unsigned int n_neigh  = d_nneigh[ idx ];   // Number of neighbors of the nearest particle

    Scalar rcutSq = rcut * rcut;
    Scalar fatrterm1 = rcutSq * rcutSq + radsubsq * radsubsq - Scalar(2.0) * rcutSq * radsumsq;
    Scalar Fcut = (fatrterm1>0) ? -32.0/3.0 * F_att / (fatrterm1 * fatrterm1) : 0;
    Fcut += F_rep / rcutSq * (kappa + 1.0/rcut) * exp(-kappa*(rcut-2.0));

    Scalar r_eff = 2.0 + h_rough + delta;
    Scalar r2 = r_eff * r_eff;
    fatrterm1 = r2 * r2 + radsubsq * radsubsq - 2.0 * r2 * radsumsq;
    Scalar Fin = -32.0/3.0 * F_att / (fatrterm1 * fatrterm1);
    Scalar dF_in = -64.0/3.0 * F_att / (fatrterm1 * fatrterm1 * fatrterm1) * (4*r_eff*r2 - 4*r_eff*radsumsq);

    r_eff = 2.0 + h_rough;
    r2 = r_eff * r_eff;
    fatrterm1 = r2 * r2 + radsubsq * radsubsq - 2.0 * r2 * radsumsq;
    Scalar Fout = -32.0/3.0 * F_att / (fatrterm1 * fatrterm1);

    Scalar h_min = 0.0;

    for (unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++) {

      // Get the current neighbor index
      unsigned int curr_neigh = d_nlist[ head_idx + neigh_idx ];

      // check if both are from rigid-body
      if((d_body_tag[curr_neigh]==-1)||(d_body_tag[curr_neigh]==d_body_tag[idx])) continue;

      Scalar4 posj = d_pos[curr_neigh];  // position
      Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );  // distance vector
      R = box.minImage(R);  //periodic BC
      Scalar  distSqr = dot(R,R);
      if(distSqr>rcutSq) continue;
      Scalar dist = sqrt( distSqr );  // Distance magnitude
      Scalar  gap1 = 2.0 - dist;  //surface gap for interparticle forces
      if(gap1>h_min) h_min=gap1;
      Scalar F_app_mag = 0.0;  //applied force magnitude

      // 1. Calculate DLVO (VDW + Electrostatic) component (always present or capped)
      if(fabs(gap1)>=delta){
        Scalar r_eff = (gap1 < 0) ? (dist + h_rough) : (2.0 + h_rough);
        Scalar r2 = r_eff * r_eff;
        Scalar fatrterm1 = r2 * r2 + radsubsq * radsubsq - Scalar(2.0) * r2 * radsumsq;
        F_app_mag += ((fatrterm1 > 0) ? -32.0/3.0 * F_att / (fatrterm1 * fatrterm1) : 0);
        F_app_mag += (F_rep / distSqr * (kappa + 1.0/dist) * exp(-kappa*(dist-2.0)) - Fcut);
      }
      else{
        Scalar h_norm = (gap1 + delta) / (2.0 * delta);
        Scalar h_norm2 = h_norm * h_norm;
        Scalar h_norm3 = h_norm * h_norm2;
        F_app_mag += ((2*h_norm3 - 3*h_norm2 + 1)*Fin + (h_norm3 - 2*h_norm2 + h_norm)*(2.0 * delta)*dF_in + (-2*h_norm3 + 3*h_norm2)*Fout);
        F_app_mag += (F_rep / distSqr * (kappa + 1.0/dist) * exp(-kappa*(dist-2.0)) - Fcut);
      }

      F_app_mag += ((gap1 > 0) ? (k_n / dist * pow(gap1,1.5)) : 0);

      // Accumulate the collision/repulsive forces
      F_x += F_app_mag * R.x;
      F_y += F_app_mag * R.y;
      F_z += F_app_mag * R.z;

      d_Velocity[6 * tidx + 0] -= F_app_mag * R.x * R.x;
      d_Velocity[6 * tidx + 1] -= F_app_mag * R.y * R.x;
      d_Velocity[6 * tidx + 2] -= F_app_mag * R.z * R.x;
      d_Velocity[6 * tidx + 3] -= F_app_mag * R.z * R.y;
      d_Velocity[6 * tidx + 4] -= F_app_mag * R.y * R.y;
      d_Velocity[6 * tidx + 5] -= F_app_mag * R.z * R.z;
    } //neighbor particle

    d_AppliedForce[ 6*tidx     ] += F_x;
    d_AppliedForce[ 6*tidx + 1 ] += F_y;
    d_AppliedForce[ 6*tidx + 2 ] += F_z;
    d_AppliedForce[ 6*tidx + 3 ] += T_x;//0.0;
    d_AppliedForce[ 6*tidx + 4 ] += T_y;//0.0;
    d_AppliedForce[ 6*tidx + 5 ] += T_z;//0.0;
    //printf("%i,%f,%f,%f,%f,%f,%f\n",idx,F_x,F_y,F_z,T_x,T_y,T_z)
    d_Velocity[6*group_size+tidx] = h_min;
  }
}

/*!
	Copy velocity computed from solving the hydrodynamic problem
	to the HOOMD velocity array

	d_vel			(output) HOOMD velocity vector
	d_Velocity		(input)  Velocity computed from hydrodynamics
	group_size		(input)  length of vectors
	d_group_members		(input)  index into vectors

*/
__global__ void Stokes_SetVelocity_kernel(
						Scalar4 *d_vel,
						Scalar4 *d_omg,
						Scalar   *d_Velocity,
						unsigned int group_size,
						unsigned int *d_group_members
						){

	// Thread idx
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	// Do work if thread is in bounds
	if (tidx < group_size) {

		Scalar4 vel,omg;
		
		unsigned int idx  = d_group_members[ tidx ];
		unsigned int idx0 = 6*idx;
		//unsigned int idx1 = 6*group_size + 5*idx;
		
		vel.x = d_Velocity[ idx0     ];
		vel.y = d_Velocity[ idx0 + 1 ];
		vel.z = d_Velocity[ idx0 + 2 ];
		omg.x = d_Velocity[ idx0 + 3 ];
		omg.y = d_Velocity[ idx0 + 4 ];
		omg.z = d_Velocity[ idx0 + 5 ];

		d_vel[ idx ] = make_scalar4( vel.x, vel.y, vel.z, 0. );
		d_omg[ idx ] = make_scalar4( omg.x, omg.y, omg.z, 0. );

	}
}

