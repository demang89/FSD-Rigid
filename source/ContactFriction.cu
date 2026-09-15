// ContactFriction.cu
// Tangential contact friction for Fast Stokesian Dynamics (HOOMD-blue plugin).
// Style matches Lubrication.cu (Fiore & Ge).
//
// Refactored to use a SINGLE 20N Fixed-Capacity Contact Journal via Frame Aging.
// Includes automated inline dead-contact cleaning to prevent slot accumulation.

#include "ContactFriction.cuh"

#include <stdio.h>
#include <math.h>
#include "hoomd/TextureTools.h"
#include <cuda_runtime.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

#define SLOT_EMPTY_KEY            0xFFFFFFFF

// ---------------------------------------------------------------------------
// Kernel: initContactTable_kernel
//
// Initializes the 20N structure. Must be run ONCE when allocating the table
// or whenever the simulation completely resets.
// ---------------------------------------------------------------------------
__global__ void initContactTable_kernel(ContactSlot *d_table, unsigned int total_slots)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= total_slots) return;

    // Set every single slot to an empty/unallocated state
    d_table[tidx].neighbor_id      = 0xFFFFFFFF; // SLOT_EMPTY_KEY
    d_table[tidx].last_seen_frame  = 0;
    d_table[tidx].xi[0]            = Scalar(0.0);
    d_table[tidx].xi[1]            = Scalar(0.0);
    d_table[tidx].xi[2]            = Scalar(0.0);
}

// ---------------------------------------------------------------------------
// Device helpers for Single-Pointer 20N structure with Aging & Eviction
// ---------------------------------------------------------------------------

__device__ inline int findSlotForUpdate(ContactSlot *d_table,
                                const unsigned int  max_contact,
                                unsigned int particle_id,
                                unsigned int neighbor_id)
{
    unsigned int base_idx = particle_id * max_contact;
    int dead_slot_idx = -1;

    for (int k = 0; k < max_contact; k++)
    {
        unsigned int slot_idx = base_idx + k;
        unsigned int slot_neighbor = d_table[slot_idx].neighbor_id;

        if (slot_neighbor == neighbor_id) return slot_idx;

        // Condition B: Keep track of slots that are completely unallocated
        if (dead_slot_idx == -1 && slot_neighbor == SLOT_EMPTY_KEY)
        {
            dead_slot_idx = slot_idx;
        }
    }
    if(dead_slot_idx==-1){
        printf("No empty slot for particle %u. Aborting.\n", particle_id);
        __trap();
    }
    return dead_slot_idx;
}



__device__ inline int findSlotForForce(ContactSlot *d_table,
                                      const unsigned int max_contact,
                                      unsigned int particle_id,
                                      unsigned int neighbor_id)
{
    unsigned int base_idx = particle_id * max_contact;

    for (int k = 0; k < max_contact; k++)
    {
        unsigned int slot_idx = base_idx + k;
        if (d_table[slot_idx].neighbor_id == neighbor_id) 
            return slot_idx;
    }
    return -1; // Return -1 to indicate contact history doesn't exist for this pair
}



// ---------------------------------------------------------------------------
// Kernel 1: ContactFriction_UpdateHistory_kernel
// ---------------------------------------------------------------------------
__global__ void ContactFriction_UpdateHistory_kernel( 
        const Scalar4      *d_pos,
        const Scalar4       *d_vel,
        const Scalar4       *d_angmom,
        unsigned int       *d_group_members,
        const unsigned int           group_size,
        int                 *d_body_tag,
        Scalar3             *d_rel_pos,
        const BoxDim        box,
        const unsigned int *d_n_neigh,
        unsigned int       *d_nlist,
        const size_t       *d_headlist,
        ContactSlot        *d_table,       // Single Pointer
	const unsigned int max_contact,
        const uint64_t  current_frame, // Bumped every timestep on host
        const Scalar        k_t,
        const Scalar        k_n,
        const Scalar        mu_f,
        const Scalar        F_att,
        const Scalar        epsq,
        const Scalar        gamma_dot,
        const Scalar        dt,
        const Scalar        radius,
	const unsigned int  *d_tag,
        const unsigned int  *d_rtag
        )
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( tidx >= group_size) return;

    Scalar L_y =  box.getL().y;
    Scalar Vinf =  L_y * gamma_dot;
    unsigned int curr_particle = d_group_members[ tidx ];
    unsigned int tagi = d_tag[ curr_particle ];
    int b_tagi = d_body_tag[curr_particle];
    Scalar3 rposi = d_rel_pos[curr_particle];

    // =======================================================================
    // FIX: INLINE DEAD CONTACT CLEANING PASS
    // Before looking at new collisions, explicitly erase contacts that 
    // were not seen in the previous frame. This prevents stale accumulation.
    // =======================================================================
    unsigned int base_idx = tagi * max_contact;
    for (int k = 0; k < max_contact; k++)
    {
        unsigned int slot_idx = base_idx + k;
        // If a slot has a valid neighbor but its frame is older than the previous frame,
        // it means it wasn't touched last frame. It is broken; delete it immediately.
        if (d_table[slot_idx].neighbor_id != SLOT_EMPTY_KEY && 
            d_table[slot_idx].last_seen_frame < current_frame - 1)
        {
            d_table[slot_idx].neighbor_id = SLOT_EMPTY_KEY;
            d_table[slot_idx].xi[0] = Scalar(0.0);
            d_table[slot_idx].xi[1] = Scalar(0.0);
            d_table[slot_idx].xi[2] = Scalar(0.0);
        }
    }
    // =======================================================================

    Scalar4 posi = d_pos[ curr_particle ];

    Scalar4 ui = d_vel[ d_rtag[b_tagi] ];
    Scalar4 wi = d_angmom[ d_rtag[b_tagi] ];

    unsigned int head_idx = d_headlist[ curr_particle ];
    unsigned int n_neigh  = d_n_neigh[ curr_particle ];

    for ( unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++ )
    {
        unsigned int curr_neigh = d_nlist[ head_idx + neigh_idx ];
        int b_tagj = d_body_tag[curr_neigh];
        if((b_tagj==-1)||(b_tagi==b_tagj)) continue;

	    unsigned int tagj = d_tag[ curr_neigh ];

        Scalar4 posj = d_pos[ curr_neigh ];

        Scalar3 R = make_scalar3( posi.x - posj.x,
                                  posi.y - posj.y,
                                  posi.z - posj.z );

        Scalar vx_fac = 0.0; 
        if(fabs(R.y)>0.5*L_y) vx_fac =  R.y > 0 ? 1.0 : -1.0;

        R = box.minImage( R );

        Scalar distSqr = dot( R, R );
        Scalar dist = sqrt( distSqr );
        Scalar overlap = Scalar(2.0)*radius - dist; //+ sqrt(epsq);

        if ( (overlap>0) && (distSqr > Scalar(0.0)) )
        {
            Scalar3 rposj = d_rel_pos[ curr_neigh ];
            Scalar3 r = make_scalar3( R.x/dist, R.y/dist, R.z/dist );

            Scalar4 uj = d_vel[ d_rtag[b_tagj] ];
            Scalar4 wj = d_angmom[ d_rtag[b_tagj] ];

            Scalar3 v_rel;
            v_rel.x = (ui.x + rposi.z*wi.y - rposi.y*wi.z) - (uj.x  + rposj.z*wj.y - rposj.y*wj.z + Vinf * vx_fac);
            v_rel.y = (ui.y + rposi.x*wi.z - rposi.z*wi.x) - (uj.y + rposj.x*wj.z - rposj.z*wj.x);
            v_rel.z = (ui.z + rposi.y*wi.x - rposi.x*wi.y) - (uj.z + rposj.y*wj.x - rposj.x*wj.y);

            Scalar vn = r.x*v_rel.x + r.y*v_rel.y + r.z*v_rel.z;

            Scalar3 w_rel;
            w_rel.x = wi.x + wj.x;
            w_rel.y = wi.y + wj.y;
            w_rel.z = wi.z + wj.z;

            Scalar3 epsrdw = make_scalar3(  r.z*w_rel.y - r.y*w_rel.z,
                                            r.x*w_rel.z - r.z*w_rel.x,
                                            r.y*w_rel.x - r.x*w_rel.y );

            Scalar3 vt = v_rel - vn*r;
            Scalar3 v_slip = vt - (radius-0.5*overlap)*epsrdw;

            Scalar F_n = k_n * pow(overlap,1.5);
	    if(epsq>0) F_n += F_att / 12.0 / epsq;
	    F_n *= mu_f;

            // Fetch slot and old history
            int slot = findSlotForUpdate(d_table, max_contact, tagi, tagj);
            Scalar3 xi = make_scalar3(d_table[slot].xi[0], d_table[slot].xi[1], d_table[slot].xi[2]);

	    //rotate xi into point of contact frame
            Scalar3 epswdxi = make_scalar3( xi.z*w_rel.y - xi.y*w_rel.z,
                                            xi.x*w_rel.z - xi.z*w_rel.x,
                                            xi.y*w_rel.x - xi.x*w_rel.y );
	    xi += (0.5 * epswdxi * dt);

            // Integrate spring
            xi.x += v_slip.x * dt;
            xi.y += v_slip.y * dt;
            xi.z += v_slip.z * dt;

            Scalar xi_mag = sqrt(xi.x*xi.x + xi.y*xi.y + xi.z*xi.z);
            if((k_t*xi_mag) > (F_n)) xi *= (F_n/(k_t*xi_mag));


            // Save and stamp the new state into the single pointer table
            d_table[slot].neighbor_id = tagj;
            d_table[slot].xi[0] = xi.x;
            d_table[slot].xi[1] = xi.y;
            d_table[slot].xi[2] = xi.z;
            d_table[slot].last_seen_frame = current_frame;

            //debug
            //printf("%u,%u,%.7f,%.7f,%.7f\n",tagi,tagj,xi.x,xi.y,xi.z);

        } // distance check
    } // neighbour loop
}

// ---------------------------------------------------------------------------
// Kernel 2: ContactFriction_Force_kernel
// ---------------------------------------------------------------------------
__global__ void ContactFriction_Force_kernel(
        Scalar             *d_Force,
        Scalar              *d_Velocity,
        const Scalar4      *d_pos,
        unsigned int       *d_group_members,
        const unsigned int group_size,
        int                 *d_body_tag,
        const BoxDim        box,
        const unsigned int *d_n_neigh,
        unsigned int       *d_nlist,
        const size_t       *d_headlist,
        ContactSlot        *d_table,       // Single Pointer
	const unsigned int max_contact,
        const uint64_t  current_frame,
        const Scalar        k_t,
        const Scalar        epsq,
        const Scalar        radius,
	const unsigned int  *d_tag
        )
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( tidx >= group_size ) return;

    unsigned int curr_particle = d_group_members[ tidx ];
    unsigned int tagi = d_tag[ curr_particle ];
    Scalar4 posi = d_pos[ curr_particle ];

    Scalar3 fi = make_scalar3( Scalar(0.0), Scalar(0.0), Scalar(0.0) );
    Scalar3 li = make_scalar3( Scalar(0.0), Scalar(0.0), Scalar(0.0) );
    Scalar stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    unsigned int head_idx = d_headlist[ curr_particle ];
    unsigned int n_neigh  = d_n_neigh[ curr_particle ];

    for ( unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++ )
    {
        unsigned int curr_neigh = d_nlist[ head_idx + neigh_idx ];

        if((d_body_tag[curr_neigh]==-1)||(d_body_tag[curr_neigh]==d_body_tag[curr_particle])) continue;

        Scalar4 posj = d_pos[ curr_neigh ];

        Scalar3 R = make_scalar3( posi.x - posj.x,
                                  posi.y - posj.y,
                                  posi.z - posj.z );
        R = box.minImage( R ); 

        Scalar distSqr = dot( R, R );
        Scalar dist = sqrt( distSqr );
        Scalar overlap = Scalar(2.0)*radius - dist;

        if ( (overlap>0) && (distSqr > Scalar(0.0)) )
        {
            unsigned int tagj = d_tag[ curr_neigh ];

            Scalar3 r = make_scalar3( R.x/dist, R.y/dist, R.z/dist );

            int slot = findSlotForForce(d_table, max_contact, tagi, tagj); 
            if(slot==-1) continue;
            Scalar3 xi = make_scalar3(d_table[slot].xi[0], d_table[slot].xi[1], d_table[slot].xi[2]); 

            // Force F_i = - k_t * xi
            Scalar3 F_on_i = make_scalar3( -k_t*xi.x, -k_t*xi.y, -k_t*xi.z );

            // Torque tau_i = -(radius * r) x F_on_i
            Scalar3 rXF = make_scalar3( r.y*F_on_i.z - r.z*F_on_i.y,
                                        r.z*F_on_i.x - r.x*F_on_i.z,
                                        r.x*F_on_i.y - r.y*F_on_i.x );

            fi.x += F_on_i.x;  fi.y += F_on_i.y;  fi.z += F_on_i.z;
            li.x -= (radius-0.5*overlap) * rXF.x;
            li.y -= (radius-0.5*overlap) * rXF.y;
            li.z -= (radius-0.5*overlap) * rXF.z;

            stress[ 0 ] -= F_on_i.x * R.x;
            stress[ 1 ] -= 0.5*(F_on_i.x * R.y + F_on_i.y * R.x);
            stress[ 2 ] -= 0.5*(F_on_i.x * R.z + F_on_i.z * R.x);
            stress[ 3 ] -= 0.5*(F_on_i.z * R.y + F_on_i.y * R.z);
            stress[ 4 ] -= F_on_i.y * R.y;
            stress[ 5 ] -= F_on_i.z * R.z;

        } // distance check
    } // neighbour loop

    unsigned int index_ = 6*tidx;
    d_Force[ index_     ] += fi.x;
    d_Force[ index_ + 1 ] += fi.y;
    d_Force[ index_ + 2 ] += fi.z;
    d_Force[ index_ + 3 ] += li.x;
    d_Force[ index_ + 4 ] += li.y;
    d_Force[ index_ + 5 ] += li.z;

    for (int c=0; c<6; c++){
        d_Velocity[ index_ + c ] += stress[c];
    }
}


// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
cudaError_t gpu_ContactFriction_RFU(
        Scalar             *d_Force,
        Scalar              *d_Velocity,
        const Scalar4      *d_pos,
        const Scalar4       *d_vel,
        const Scalar4       *d_angmom,
        unsigned int       *d_group_members,
        const unsigned int  group_size,
        int                 *d_body_tag,
        Scalar3             *d_rel_pos,
        const BoxDim&       box,
        const unsigned int *d_n_neigh,
        unsigned int       *d_nlist,
        const size_t       *d_headlist,
        ContactSlot        *d_table,
	const unsigned int max_contact,
        const uint64_t  current_frame, 
        const Scalar        k_t,
        const Scalar        k_n,
        const Scalar        mu_f,
        const Scalar        F_att,
        const Scalar        epsq,
        const Scalar        gamma_dot,
        const Scalar        dt,
        const Scalar        radius,
        KernelData          *ker_data,
	const unsigned int  *d_tag,
        const unsigned int  *d_rtag
        )
{
    dim3 grid    = ker_data->particle_grid;
    dim3 threads = ker_data->particle_threads;

    // Step 1: Clean broken contacts inline, check current overlaps, integrate, and write back
    ContactFriction_UpdateHistory_kernel<<<grid, threads>>>(
            d_pos, d_vel, d_angmom, d_group_members, group_size, d_body_tag, d_rel_pos, box,
            d_n_neigh, d_nlist, d_headlist, d_table, max_contact, current_frame,
            k_t, k_n, mu_f, F_att, epsq, gamma_dot, dt, radius, d_tag, d_rtag);

    cudaDeviceSynchronize();

    // Step 2: Calculate forces using the updated state tracking layout
    ContactFriction_Force_kernel<<<grid, threads>>>(
            d_Force, d_Velocity, d_pos, d_group_members, group_size, d_body_tag,
            box, d_n_neigh, d_nlist, d_headlist, d_table, max_contact, current_frame,
            k_t, epsq, radius, d_tag);

    return cudaSuccess;
}
