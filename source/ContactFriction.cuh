// ContactFriction.cuh
// Header for Lees-Edwards aware tangential contact friction kernel.
// Style matches Lubrication.cuh (Fiore & Ge).

#ifndef __CONTACT_FRICTION_CUH__
#define __CONTACT_FRICTION_CUH__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/ParticleData.cuh"
#include <cuda_runtime.h>
#include "DataStruct.h"

// ---------------------------------------------------------------------------
// gpu_ContactFriction_RFU
//
// Compute tangential contact friction forces and ADD them to d_Force.
//
// Lees-Edwards: d_Velocity contains disturbance velocities U' = U - U^inf.
// The shear-rate gamma_dot is used to reconstruct the full relative velocity
// across y-periodic images: v_rel += gamma_dot * R_y_img * xhat.
//
// Allocation (once in integrator constructor):
//   unsigned int table_size = 20 * group_size;  // safe for dense suspensions
//   cudaMalloc(&m_d_table, table_size * sizeof(ContactSlot));
//   cudaMemset(m_d_table, 0, table_size * sizeof(ContactSlot));
//
// Call each timestep after the lubrication force kernel:
//   gpu_ContactFriction_RFU(
//       d_Force, d_Velocity, d_pos, d_group_members, group_size,
//       box, d_n_neigh, d_nlist, d_headlist,
//       m_d_table, table_size,
//       m_k_t, m_k_n, m_mu_f,
//       m_gamma_dot,              // shear rate (0 if no LE / quiescent)
//       m_dt, m_radius,
//       2.001 * m_radius,         // rcontact
//       block_size );
//
// Parameters:
//   d_Force        [in/out] 6*N generalized force (force + torque)
//   d_Velocity     [in]     6*N disturbance velocity U' = U - U^inf
//   d_pos          [in]     N  Scalar4 positions
//   d_group_members[in]     particle index map
//   group_size              number of particles
//   box                     simulation box (minImage includes LE affine shift)
//   d_n_neigh, d_nlist, d_headlist   full neighbour list
//   d_table        [in/out] hash table of ContactSlot (nlist-rebuild safe)
//   table_size              allocated slots (>= 2 * max simultaneous contacts)
//   k_t                     tangential spring stiffness  [force/length]
//   k_n                     normal   spring stiffness    [force/length]
//   mu_f                    Coulomb friction coefficient [dimensionless]
//   gamma_dot               shear rate (set to 0 for quiescent / non-LE runs)
//   dt                      timestep
//   radius                  particle radius (monodisperse)
//   rcontact                contact shell cutoff (e.g. 2.001 * radius)
//   block_size              CUDA threads per block (e.g. 256)
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
        );

__global__ void initContactTable_kernel(ContactSlot *d_table, unsigned int total_slots);

#endif // __CONTACT_FRICTION_CUH__
