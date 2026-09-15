// This file is part of the PSEv3 plugin, released under the BSD 3-Clause License
//
// Andrew Fiore


#include "Precondition.cuh"
#include "Lubrication.cuh"

#include "Helper_Debug.cuh"
#include "Helper_Precondition.cuh"

#include "rcm.hpp"

#include <stdio.h>
#include <math.h>
#include "hoomd/TextureTools.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/version.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse.h>
#include <cusolverSp.h>

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! Command to convert Scalars or doubles to integers
#ifdef SINGLE_PRECISION
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

__global__ void Precondition_AddGeometricReg_kernel(
		const int     N_agg,
		const int     nb,
		const unsigned int    *d_rtag,
		const Scalar3 *d_rel_pos,
		const int    *d_HasNeigh,
		const int    *d_L_RowPtr,
		const int    *d_L_ColInd,
		Scalar       *d_L_Val, //output
		const Scalar  sigma // regularization strength
){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( tidx >= N_agg ) return;

    // Only apply to isolated aggregates
    //if ( d_HasNeigh[ tidx ] ) return;

    // Build the 6×6 matrix Σ_i G_i^T G_i in local registers
    Scalar reg[6][6] = {0};

    int start = nb * tidx + N_agg;
    for ( int s = 0; s < nb; ++s ) {
        const int    sphere_i = d_rtag[ start+s ];
        const Scalar3 b       = d_rel_pos[ sphere_i ];

        // [b]× in JO left-handed convention (negated standard)
        //const Scalar bx[3][3] = {
        //    {  0.0,   b.z,   -b.y },
        //    {  -b.z,   0.0,   b.x },
        //    { b.y,   -b.x,   0.0  }
        //};

        // Top-left 3×3: += I
        for ( int k = 0; k < 3; ++k )
            reg[k][k] += 1.0;

        //// Top-right 3×3: += -[b]×  (= +standard [b]× since JO negated)
        //for ( int k = 0; k < 3; ++k )
        //    for ( int l = 0; l < 3; ++l )
        //        reg[k][3+l] += bx[k][l];

        //// Bottom-left 3×3: += [b]×  (JO)
        //for ( int k = 0; k < 3; ++k )
        //    for ( int l = 0; l < 3; ++l )
        //        reg[3+k][l] += -bx[k][l];

        // Bottom-right 3×3: += -[b]×[b]× + I
        // -[b]×[b]× = b b^T - |b|^2 I
        Scalar bsq = b.x*b.x + b.y*b.y + b.z*b.z;
        Scalar bvec[3] = { b.x, b.y, b.z };
        for ( int k = 0; k < 3; ++k ) {
            for ( int l = 0; l < 3; ++l ) {
                reg[3+k][3+l] += bvec[k]*bvec[l];  // b b^T
            }
            //reg[3+k][3+k] += 1.33333333f - bsq;            // (1 - |b|^2) on diagonal
	    reg[3+k][3+k] += (bsq > 4.0/3.0) ? bsq : 4.0/3.0;

        }
    }

    // Add sigma * reg[k][l] to the corresponding entry in the CSR matrix
    const int row_base = 6 * tidx;

    for ( int k = 0; k < 6; ++k ) {
        const int row = row_base + k;
        for ( int jj = d_L_RowPtr[row]; jj < d_L_RowPtr[row+1]; ++jj ) {
            const int col = d_L_ColInd[jj];
            // Only touch entries in the self-block (col in [row_base, row_base+6))
            if ( col >= row_base && col < row_base + 6 ) {
                const int l = col - row_base;
                d_L_Val[jj] += sigma * reg[k][l];
            }
        }
    }
}



__global__ void print_neighbor_list_kernel(const unsigned int* nlist,
                                           const unsigned int* head_list,
                                           const unsigned int* n_neigh,
                                           const unsigned int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int head = head_list[i];
    unsigned int nn = n_neigh[i];

    printf("(p%u:ngh=%u,head=%u,%u,%u)\n", i, nn, head, nlist[head], nlist[head + 1]);

    //for (unsigned int k = 0; k < nn; ++k)
    //{
    //    unsigned int j = nlist[head + k];
    //    printf("%u,", j);
    //}
    //printf(")\n");
}


void printCSRMatrix(int num_rows, int nnz,
                    int* d_csrRowPtr, int* d_csrColInd, Scalar* d_csrVal) {
    // Allocate host memory
    int* h_rowPtr = new int[num_rows+1];
    int* h_colInd = new int[nnz];
    Scalar* h_val = new Scalar[nnz];

    // Copy from device to host
    cudaMemcpy(h_rowPtr, d_csrRowPtr, (num_rows+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colInd, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val, d_csrVal, nnz * sizeof(Scalar), cudaMemcpyDeviceToHost);

    // Print in triplet format (row, col, val)
    for (int row = 0; row < num_rows; ++row) {
        int start = h_rowPtr[row];
        int end = h_rowPtr[row + 1];
	if((end-start)>1){
        for (int idx = start; idx < end; ++idx) {
	    printf("(%d,%d,%.2f),",row,h_colInd[idx],h_val[idx]);
            //std::cout << "(" << row << "," << h_colInd[idx] << "," << h_val[idx] << ") ";
	}
        }
    }
    std::cout << "\n";

    // Clean up
    delete[] h_rowPtr;
    delete[] h_colInd;
    delete[] h_val;
}

/*
	This file defines functions required to build the Preconditioner
	for the lubrication and saddle point problems,
*/


/*!
	Get the Pruned neighborlist (i.e. contains only particles that are actually within
	the cutoff) and sort it by particle index.

	Function 1 -- Get the pruned number of neighbors 

	d_nneigh_pruned		(output) number of neighbors within the preconditioner cutoff
	group_size		(input)  number of particles
	d_pos			(input)  particle positions
	box			(input)  periodic box information
	d_group_members		(input)  array of particle indices within the integration group
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nneigh_less		(input)  list of number of neighbors for each particle with index less than the particle's index
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	rp			(input)  cutoff radius for the preconditioner

*/
__global__ void Precondition_GetPrunedNneigh_kernel( 	
							unsigned int *d_nneigh_pruned, 
							const unsigned int N_agg,
							const unsigned int nb,
							const Scalar4 *d_pos,
							const BoxDim box,
							const unsigned int *d_rtag,
							const int *d_body_tag,
							const unsigned int *d_nneigh, 
							const unsigned int *d_nlist, 
							const size_t *d_headlist,
							const Scalar rp
							){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx >= N_agg) return;

	// Cutoff radius squared
	Scalar rpsq = rp * rp;

	const int MAX_AGG_NEIGH = 13;
	int found[MAX_AGG_NEIGH];
	found[0] = tidx;
	int n_found = 1;

	int start = nb * tidx + N_agg;
	for (int s = 0; s < nb; ++s) {
		int sphere_i = d_rtag[ start+s ];
		Scalar4 posi = d_pos[ sphere_i ];
		unsigned int head_idx = d_headlist[ sphere_i ]; // Location in head array for neighbors of current particle
		unsigned int nneigh = d_nneigh[ sphere_i ];   // Number of neighbors of the nearest particle
		for ( int ii = 0; ii < nneigh; ii++ ){
			
			// Get the current neighbor index
			unsigned int curr_neigh = d_nlist[ head_idx + ii ];
			int agg_j = d_body_tag[curr_neigh];
			if((agg_j==tidx)||(agg_j==-1)) continue;

			// Position of current neighbor
			Scalar4 posj = d_pos[ curr_neigh ];
			
			// Distance between current particle and neighbor
			Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
			R = box.minImage(R);
			Scalar distSqr = dot(R,R);
			if(distSqr > rpsq ) continue;
			// Check if agg_j is already recorded
			bool dup = false;
			for (int f = 0; f < n_found; ++f)
				if (found[f] == agg_j) { dup = true; break; }
			if (!dup && n_found < MAX_AGG_NEIGH)
				found[n_found++] = agg_j;
		}//loop over neighbors

	}//loop over beads

	d_nneigh_pruned[ tidx ] = n_found;

	//if(n_found>1) printf("(%d,%d)",tidx,n_found); //debug
}

/*!
	Get the Pruned neighborlist (i.e. contains only particles that are actually within
	the cutoff) and sort it by particle index.

	Function 2 -- Get the pruned and sorted neighborlist for particles within the
			preconditioner cutoff

	Use an insertion sort -- This will be good as long as each particles don't have
		too many neighbors in the list. This will be true for the lubrication
		tensor because the cutoff radius is 4 particle radii. 
	
	d_nneigh_pruned		(input)		number of neighbors within the preconditioner cutoff
	d_nlist_pruned		(input/output) 	sorted neighborlist pruned to neighborlist cutoff
	d_headlist_pruned	(input)		headlist for each particle location in the neighborlist
	d_nneigh_less		(output)	number of particles with index less than each particle
	group_size		(input)		number of particles
	d_pos			(input)		particle positions
	box			(input)		periodic box information
	d_group_members		(input)		array of particle indices within the integration group
	d_nneigh		(input)		list of number of neighbors for each particle
	d_nneigh_less		(input)		list of number of neighbors for each particle with index less than the particle's index
	d_nlist			(input)		neighborlist array
	d_headlist		(input)		indices into the neighborlist for each particle
	rp			(input)		cutoff radius for the preconditioner
	
*/
__global__ void Precondition_GetPrunedNlist_kernel( 	
							unsigned int *d_nneigh_pruned, 
							unsigned int *d_nlist_pruned,
							unsigned int *d_headlist_pruned,
							unsigned int *d_nneigh_less, 
							const unsigned int N_agg,
							const unsigned int nb,
							const Scalar4 *d_pos,
							const BoxDim box,
							const unsigned int *d_rtag,
							const int *d_body_tag,
							const unsigned int *d_nneigh, 
							const unsigned int *d_nlist, 
							const size_t *d_headlist,
							const Scalar rp
							){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx >= N_agg) return;

	// Cutoff radius squared
	Scalar rpsq = rp * rp;

        // Particle info for this thread, pruned
        unsigned int phead_idx = d_headlist_pruned[ tidx ]; // Location in head array for neighbors of current particle
        unsigned int pnneigh   = d_nneigh_pruned[ tidx ];   // Number of neighbors of the nearest particle

        const int MAX_AGG_NEIGH = 13;
        int found[MAX_AGG_NEIGH];
	found[0] = tidx;
        int n_found = 1;
	int counter = 1;
	d_nlist_pruned[ phead_idx ] = tidx;

        int start = nb * tidx + N_agg;
        for (int s = 0; s < nb; ++s) {
                int sphere_i = d_rtag[ start+s ];
                Scalar4 posi = d_pos[ sphere_i ];
                unsigned int head_idx = d_headlist[ sphere_i ]; // Location in head array for neighbors of current particle
                unsigned int nneigh = d_nneigh[ sphere_i ];   // Number of neighbors of the nearest particle

                for ( int ii = 0; ii < nneigh; ii++ ){

                        // Get the current neighbor index
                        unsigned int curr_neigh = d_nlist[ head_idx + ii ];
			int agg_j = d_body_tag[curr_neigh];
			if((agg_j==tidx)||(agg_j==-1)) continue;

                        // Position of current neighbor
                        Scalar4 posj = d_pos[ curr_neigh ];

                        // Distance between current particle and neighbor
                        Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
                        R = box.minImage(R);
                        Scalar distSqr = dot(R,R);
                        if(distSqr > rpsq ) continue;
                        // Check if agg_j is already recorded
                        bool dup = false;
                        for (int f = 0; f < n_found; ++f)
                                if (found[f] == agg_j) { dup = true; break; }
                        if (!dup && n_found < MAX_AGG_NEIGH)
			{
				found[n_found++] = agg_j;
				d_nlist_pruned[ phead_idx + counter ] = agg_j;
				counter++;
			}
		}//loop over neighbors
	}//loop over beads

	// Pointer to proper location within nlist
	unsigned int *A = &d_nlist_pruned[ phead_idx ];

	// Sort the neighbors using an insertion sort
	int jj, key;
	for ( int ii = 1; ii < pnneigh; ++ii ){

		// Get the current value
		key = A[ ii ];

		// Move elements that are greater than the key to one position ahead of their
		// current position
		jj = ii - 1;
		while( jj >= 0 && A[ jj ] > key ){
			A[ jj + 1 ] = A[ jj ];
			jj--;
		}
		A[ jj + 1 ] = key;
	}
	
	// Figure out how many of the neighbors have indices less than the current particle
	int nless = 0;
	for ( int ii = 0; ii < pnneigh; ++ii ){		
		if ( A[ii] < tidx ){
			nless++;
		}
	}
	d_nneigh_less[ tidx ] = nless;

	//printf("(%d,%d)",tidx,nless);
	// Clear pointers
	A = NULL;
}

/*!

	Wrap the functions to compute and sort the pruned neighborlist
	
	d_pos			(input)		particle positions
	d_group_members		(input)		array of particle indices within the integration group
	group_size		(input)		number of particles
	box			(input)		periodic box information
	res_data		(input/output)	structure containing lubrication calculation information, including neighborlist
	ker_data		(input)		structure containing kernel launch information
	
*/
void Precondition_PruneNeighborList(
					Scalar4 *d_pos,
					const unsigned int *d_rtag,
					BoxDim box,
					ResistanceData *res_data,
					KernelData *ker_data
					){

	// Kernel Information
	dim3 grid = ker_data->rigid_grid;
	dim3 threads = ker_data->rigid_threads;

	unsigned int N_agg = res_data->N_rigid;
	unsigned int nb = res_data->m_nb;

	// Get pruned number of neighbors (rp < rlub)
	Precondition_GetPrunedNneigh_kernel<<<grid,threads>>>( 	
								res_data->nneigh_pruned, 
								N_agg,
								nb,
								d_pos,
								box,
								d_rtag,
								res_data->body_tag,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist,
								res_data->rp
								);

	// Compute the pruned headlist with an inclusive scan (zhoge ???)
	int zero = 0;
	cudaMemcpy( res_data->headlist_pruned, &zero, sizeof(int), cudaMemcpyHostToDevice );

	thrust::device_ptr<unsigned int> i_thrustptr = thrust::device_pointer_cast( res_data->nneigh_pruned );
    	thrust::device_ptr<unsigned int> o_thrustptr = thrust::device_pointer_cast( (res_data->headlist_pruned)+1 );
	thrust::inclusive_scan( i_thrustptr, i_thrustptr + N_agg, o_thrustptr );
	
	// Build the (sorted) pruned neighbor list
	Precondition_GetPrunedNlist_kernel<<<grid,threads>>>( 	
								res_data->nneigh_pruned, 
								res_data->nlist_pruned,
								res_data->headlist_pruned,
								res_data->nneigh_less,
								N_agg,
								nb,
								d_pos,
								box,
								d_rtag,
								res_data->body_tag,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist,
								res_data->rp
								);

}

/*!
	Figure out how many blocks of entries in the resistance tensor preconditioner
	that each particle has, i.e. total number of Nonzero Entries Per Particle (NEPP)

	group_size		(input)  length of vector d_nneigh
	d_group_members		(input)  array of particle indices within the integration group
	d_nneigh_pruned		(input)  list of number of neighbors for each particle in the pruned list
	d_nlist_pruned		(input)  pruned neighborlist array
	d_headlist_pruned	(input)  indices into the pruned neighborlist for each particle
	d_NEPP			(output) Number of non-zero entries per particle
	
*/
__global__ void Precondition_NEPP_kernel( 	
						unsigned int N_agg,
						const unsigned int *d_nneigh_pruned, 
						const unsigned int *d_nlist_pruned, 
						const unsigned int *d_headlist_pruned, 
						unsigned int *d_NEPP
						){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx >= N_agg) return;

	// Number of neighbors for current particle.	
	unsigned int nneigh = d_nneigh_pruned[ tidx ]; // Number of neighbors of the nearest particle

	// Pruned neighborlist contains SELF ID as well, so if a particle has zero neighbors,
	// nneigh = 1. Also need to save space for the self block.
	int ne1 = ( nneigh > 1 ) ? ( ( 9 + 9 ) * ( nneigh ) ) : ( 3 );
	int ne2 = ( nneigh > 1 ) ? ( ( 9 + 9 ) * ( nneigh ) ) : ( 3 );

	// Write out
	d_NEPP[              tidx ] = ne1;
	d_NEPP[ N_agg + tidx ] = ne2;
	//printf("(%d,%d,%d)",tidx,ne1,ne2);
}

/*!

	Figure out whether a particle has neighbors within the lubrication cutoff
	Return 1 or 0 for each particle.

	d_HasNeigh		(output) list of whether particle has neighbors in the lubrication cutoff
	group_size		(input)  number of particles in the group
	d_pos			(input)  particle positions
	box			(input)  periodic box information
	d_group_members		(input)  array of particle indices
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	rlub			(input)  cutoff radius for lubrication interactions

*/
__global__ void Precondition_HasNeigh_kernel( 	
						int *d_HasNeigh,
						const unsigned int N_agg,
						const unsigned int nb,
						const Scalar4 *d_pos,
						const BoxDim box,
						const unsigned int *d_rtag,
						const int *d_body_tag,
						const unsigned int *d_nneigh, 
						const unsigned int *d_nlist, 
						const size_t *d_headlist,
						const Scalar rlub
						){

	// Current thread index	
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Value of summation variables for the current thread
	if ( tidx >= N_agg) return;

	// Cutoff radius squared
	Scalar rlubsq = rlub * rlub;

        int start = nb * tidx + N_agg;
        for (int s = 0; s < nb; ++s) {
                int sphere_i = d_rtag[ start+s ];
                Scalar4 posi = d_pos[ sphere_i ];
                unsigned int head_idx = d_headlist[ sphere_i ]; // Location in head array for neighbors of current particle
                unsigned int nneigh = d_nneigh[ sphere_i ];   // Number of neighbors of the nearest particle
                for ( int ii = 0; ii < nneigh; ii++ ){

                        // Get the current neighbor index
                        unsigned int curr_neigh = d_nlist[ head_idx + ii ];
			int agg_j  = d_body_tag[curr_neigh];
			if((agg_j==tidx)||(agg_j==-1)) continue;

                        // Position of current neighbor
                        Scalar4 posj = d_pos[ curr_neigh ];

                        // Distance between current particle and neighbor
                        Scalar3 R = make_scalar3( posi.x - posj.x, posi.y - posj.y, posi.z - posj.z );
                        R = box.minImage(R);
                        Scalar distSqr = dot(R,R);
                        if(distSqr < rlubsq ){
				d_HasNeigh[ tidx ] = 1;
				return;
			}
                }//loop over neighbors

        }//loop over beads
}



/*!
	Pre-processing of data arrays in order to construct the sparse representation of the
	lubrication resistance tensor for the preconditioner

	d_group_members		(input)  array of particle indices within the group
	group_size		(input)  number of particles
	nnz			(input)  total number of non-zero elements within the preconditioner
	d_nneigh_pruned		(input)  list of pruned number of neighbors for each particle
	d_nlist_pruned		(input)  pruned neighborlist array
	d_headlist_pruned	(input)  indices into the pruned neighborlist for each particle
	d_NEPP			(output) Number of non-zero entries per particle	
	d_offset		(output) current particle's offsets into the output arrays
	grid			(input)  Grid for CUDA kernel launch
	threads			(input)  Threads for CUDA kernel launch
*/
void Precondition_PreProcess(
				int group_size, 
				int &nnz,
				const unsigned int *d_nneigh_pruned, 
				const unsigned int *d_nlist_pruned, 
				const unsigned int *d_headlist_pruned, 
				unsigned int *d_NEPP,
				unsigned int *d_offset,
				dim3 grid,
				dim3 threads
				){
	
	// Figure out the number of non-zero elements per particle (NEPP)
	Precondition_NEPP_kernel<<< grid, threads >>>(
							group_size,
							d_nneigh_pruned,
							d_nlist_pruned,
							d_headlist_pruned,
							d_NEPP
							);

	// First particle has offset of zero
	int zero = 0;
	cudaMemcpy( d_offset, &zero, sizeof(int), cudaMemcpyHostToDevice );
	
	// Add number of non-zero entries A,B,C for each particle 
	// ( This is needed for particle offset, but need A/B distinct 
	//   from B/C for the indexing later on )
	Precondition_AddInt_kernel<<<grid,threads>>>( &d_NEPP[0], &d_NEPP[group_size], &d_offset[1], 1, 1, group_size-1 );

	//	
	// Use THRUST to get get the cumulative sum of the numbers of entries per particle for each block
	//
	
	// Thrust device pointers for reductions
	thrust::device_ptr<unsigned int> i_thrustptr;
        thrust::device_ptr<unsigned int> o_thrustptr;
	
	// Wrap raw pointers in Thrust device pointers 
	i_thrustptr = thrust::device_pointer_cast( d_offset + 1 );	
	o_thrustptr = thrust::device_pointer_cast( d_offset + 1 );
	
	// Do the scan (cumulative sum) for the RFU mobility tensor
	thrust::inclusive_scan( i_thrustptr, i_thrustptr + (group_size-1), o_thrustptr );

	// Figure out the number of non-zero entries in each array
	int scan, end;
	cudaMemcpy( &scan, &d_offset[ group_size-1 ], sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( &end,  &d_NEPP[   group_size-1 ], sizeof(int), cudaMemcpyDeviceToHost );
	
	nnz = scan + end;

	// Have to do again for BC part of first nnz
	cudaMemcpy( &end, &d_NEPP[ 2*group_size-1 ], sizeof(int), cudaMemcpyDeviceToHost );
	nnz += end;
	
}

/*! 
	Build the preconditioner for the RFU Lubrication Tensor ( ALL PARTICLES SAME SIZE ) -- give one thread per particle 
	
	THIS VERSION OF THE FUNCTION STORES THE FULL RESISTANCE TENSOR (NOT JUST THE LOWER HALF)
		Reason: Not all cusparse operations are defined for symmetric matrices, so the 
			full, general matrices have to be used instead.
	Sparse matrix storage format is COO. 

	group_size		(input)  Number of particles
	d_group_members		(input)  array of particle indices
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nneigh_less		(input)  number of neighbors with index less than current particle
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	d_NEPP			(input)  Number of non-zero entries per particle
	d_offset		(input)  current particle's offsets into the output arrays
	d_pos			(input)  particle positions
	box			(input)  simulation box information
	d_ResTable_dist		(input)  distances for which the resistance function has been tabulated
	d_ResTable_vals		(input)  tabulated values of the resistance tensor
	table_dr		(input)  table discretization (in log units)
	d_L_RowInd		(output) COO row indices
	d_L_ColInd		(output) COO col indices
	d_L_Val			(output) COO vals
	rp			(input)  preconditioner cutoff radius

*/
__global__ void Precondition_RFU_kernel(
					const unsigned int N_agg,
				        const unsigned int nb,	
					unsigned int *d_rtag,
					Scalar3 *d_rel_pos,
					const unsigned int *d_nneigh_pruned, 
					unsigned int *d_nneigh_less, 
					unsigned int *d_nlist_pruned, 
					const unsigned int *d_headlist_pruned,
					unsigned int *d_NEPP,
					unsigned int *d_offset, 
					Scalar4 *d_pos,
					BoxDim box,
					const Scalar *d_ResTable_dist,
					const Scalar *d_ResTable_vals,
					const Scalar table_min,
					const Scalar table_dr,
					int   *d_L_RowInd,
					int   *d_L_ColInd,
					Scalar *d_L_Val,
					const Scalar rp
					){

  // Index for current thread 
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	
  // Check that thread is within bounds, and only do work if so	
  if ( tidx >= N_agg ) return;

  // Square of the cutoff radius
  Scalar rpsq = rp * rp;

  unsigned int nneigh_less = d_nneigh_less[ tidx ]; 	// Number of neighbors with index less than current particle
  unsigned int head_idx = d_headlist_pruned[ tidx ];
  unsigned int nneigh = d_nneigh_pruned[ tidx ];

  // Offset information for current particle
  unsigned int offset_particle = d_offset[ tidx ];
  unsigned int offset_BC = d_NEPP[ tidx ];

  int flag = 0; // flag to write out self elements only once

  int start_a = nb * tidx + N_agg;

  for ( unsigned int neigh_idx = 0; neigh_idx < nneigh; ++neigh_idx ) {

          unsigned int curr_neigh_agg = d_nlist_pruned[ head_idx + neigh_idx ];
          if (  curr_neigh_agg == tidx ) continue;

  	  // -----------------------------------------------------------------------
  	  //  Self-block register accumulators.
  	  //  Accumulated across ALL β iterations, written once after the outer loop.
  	  // -----------------------------------------------------------------------
  	  Scalar Ã_s[3][3]   = {0};   // Ã  self (FU×FU)
  	  Scalar BtT_s[3][3] = {0};   // B̃^T self (FU×TΩ) — carries JO self sign on YB
  	  Scalar Bt_s[3][3]  = {0};   // B̃  self (TΩ×FU)
  	  Scalar C_s[3][3]   = {0};   // C̃  self (TΩ×TΩ)

          // Cross-block accumulators (used only when !is_self)
          Scalar Ã[3][3]   = {0};
          Scalar BtT[3][3] = {0};
          Scalar Bt[3][3]  = {0};
          Scalar C[3][3]   = {0};

          int start_b = nb * curr_neigh_agg + N_agg;

    	  for (int sa = 0; sa < nb; ++sa) {
              int sphere_i = d_rtag[ start_a+sa ];
	      Scalar4 posi = d_pos[ sphere_i ];
	      Scalar3 bi= d_rel_pos[sphere_i];
	      const Scalar bix[3][3] = {
                          {  0.0,    bi.z,   -bi.y },
                          {  -bi.z,   0.0,    bi.x },
                          { bi.y,   -bi.x,   0.0   }
                      };

              for (int sb = 0; sb < nb; ++sb) {
                  int sphere_j = d_rtag[ start_b+sb ];
                  Scalar4 posj = d_pos[ sphere_j ];
                  // R = r_j − r_i  (i→j, matches sphere code convention)
                  Scalar3 R = make_scalar3( posj.x - posi.x,
                                            posj.y - posi.y,
                                            posj.z - posi.z );
                  R = box.minImage(R);
                  const Scalar distSqr = dot(R, R);
                  if ( distSqr >= rpsq ) continue;

                  // ---- Look up scalar resistance functions ------------------
                  const Scalar dist = sqrt( distSqr );

                  Scalar XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12;

	          if ( dist <= (2.0+0.00010) ){
	            // In Stokes_ResistanceTable.cc, h_ResTable_dist.data[232] = 2.000997;
	            // Table is strided by 22
	            int i_regl = 0*22; //lubrication regularization (due to roughness) 
	            XA11 = d_ResTable_vals[ i_regl + 0 ];
	            XA12 = d_ResTable_vals[ i_regl + 1 ];
	            YA11 = d_ResTable_vals[ i_regl + 2 ];
	            YA12 = d_ResTable_vals[ i_regl + 3 ];
	            YB11 = d_ResTable_vals[ i_regl + 4 ];
	            YB12 = d_ResTable_vals[ i_regl + 5 ];
	            XC11 = d_ResTable_vals[ i_regl + 6 ];
	            XC12 = d_ResTable_vals[ i_regl + 7 ];
	            YC11 = d_ResTable_vals[ i_regl + 8 ];
	            YC12 = d_ResTable_vals[ i_regl + 9 ];
	          }
	          else {

	              // Get the index of the nearest entry below the current distance in the distance array
	              // NOTE: Distances are logarithmically spaced in the tabulation
	              int ind = int(log10( ( dist - 2.0 ) /  table_min ) / table_dr);
	            					
	              // Get the values from the distance array for interpolation
	              Scalar dist_lower = d_ResTable_dist[ ind ];
	              Scalar dist_upper = d_ResTable_dist[ ind + 1 ];
	            		
	              // Read the scalar resistance coefficients from the array (lower and upper values 
	              // for interpolation)
	              //
	              // Table is strided by 22
	              Scalar XA11_lower = d_ResTable_vals[ 22 * ind + 0 ];
	              Scalar XA12_lower = d_ResTable_vals[ 22 * ind + 1 ];
	              Scalar YA11_lower = d_ResTable_vals[ 22 * ind + 2 ];
	              Scalar YA12_lower = d_ResTable_vals[ 22 * ind + 3 ];
	              Scalar YB11_lower = d_ResTable_vals[ 22 * ind + 4 ];
	              Scalar YB12_lower = d_ResTable_vals[ 22 * ind + 5 ];
	              Scalar XC11_lower = d_ResTable_vals[ 22 * ind + 6 ];
	              Scalar XC12_lower = d_ResTable_vals[ 22 * ind + 7 ];
	              Scalar YC11_lower = d_ResTable_vals[ 22 * ind + 8 ];
	              Scalar YC12_lower = d_ResTable_vals[ 22 * ind + 9 ];
	            
	              Scalar XA11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 0 ];
	              Scalar XA12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 1 ];
	              Scalar YA11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 2 ];
	              Scalar YA12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 3 ];
	              Scalar YB11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 4 ];
	              Scalar YB12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 5 ];
	              Scalar XC11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 6 ];
	              Scalar XC12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 7 ];
	              Scalar YC11_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 8 ];
	              Scalar YC12_upper = d_ResTable_vals[ 22 * ( ind + 1 ) + 9 ];
	            	
	              // Linear interpolation of the Table values
	              Scalar fac = ( dist - dist_lower  ) / ( dist_upper - dist_lower );
	            
	              XA11 = XA11_lower + ( XA11_upper - XA11_lower ) * fac;
	              XA12 = XA12_lower + ( XA12_upper - XA12_lower ) * fac;
	              YA11 = YA11_lower + ( YA11_upper - YA11_lower ) * fac;
	              YA12 = YA12_lower + ( YA12_upper - YA12_lower ) * fac;
	              YB11 = YB11_lower + ( YB11_upper - YB11_lower ) * fac;
	              YB12 = YB12_lower + ( YB12_upper - YB12_lower ) * fac;
	              XC11 = XC11_lower + ( XC11_upper - XC11_lower ) * fac;
	              XC12 = XC12_lower + ( XC12_upper - XC12_lower ) * fac;
	              YC11 = YC11_lower + ( YC11_upper - YC11_lower ) * fac;
	              YC12 = YC12_lower + ( YC12_upper - YC12_lower ) * fac;
	            }
	            		
	           // Geometric quantities ----- zhoge: THIS PART MAY NEED CORRECTION!!!
	           //
	           // levi-civita is left-handed because JO system is left-handed
	           Scalar rx = R.x / dist;
	           Scalar ry = R.y / dist;
	           Scalar rz = R.z / dist;
	           Scalar rhat[3] = { rx, ry, rz };
	           Scalar epsr[3][3] = { { 0.0, rz, -ry }, 
	           		      { -rz, 0.0, rx }, 
	           		      {  ry, -rx, 0.0} };

                   Scalar3 bj= d_rel_pos[sphere_j];
                   const Scalar bjx[3][3] = {
                               {  0.0,    bj.z,   -bj.y },
                               {  -bj.z,   0.0,    bj.x },
                               { bj.y,   -bj.x,   0.0   }
                           };
	
	           //
	           // Compute FU Block
	           //
	           for ( int k = 0; k < 3; ++k ){		
		     for ( int l = 0; l < 3; ++l ) {

			const Scalar rkrl = rhat[k] * rhat[l];
                        const Scalar Imrkrl  = ((k==l) ? 1.0:0.0) - rkrl;

                        // Sphere-sphere blocks at (k,l)
                        const Scalar A12kl  = XA12*rkrl + YA12*Imrkrl;
                        const Scalar Bt12kl = YB12 * epsr[k][l];   // B^T_{kl} = YB*epsr[k][l]
                        const Scalar C12kl  = XC12*rkrl + YC12*Imrkrl;

                        // ---- Cross block ----
                        // Ã_{kl} = A12_{kl}
                        Ã[k][l] += A12kl;
                    
                        // B̃^T_{kl} = Σ_m A12_{km} bjx[m][l] + B^T12_{kl}
                        Scalar BtT_val = Bt12kl;
                        for ( int m = 0; m < 3; ++m ) {
                            const Scalar rkrm   = rhat[k]*rhat[m];
                            const Scalar Imrkrm = (k==m ? 1.0:0.0) - rkrm;
                            const Scalar A12km  = XA12*rkrm + YA12*Imrkrm;
                            BtT_val += A12km * bjx[m][l];
                        }
                        BtT[k][l] += BtT_val;
                    
                        // B̃_{kl} = -Σ_m bix[k][m] A12_{ml} + B12_{kl}
                        Scalar Bt_val = Bt12kl;
                        for ( int m = 0; m < 3; ++m ) {
                            const Scalar rmrl   = rhat[m]*rhat[l];
                            const Scalar Imrmrl = (m==l ? 1.0:0.0) - rmrl;
                            const Scalar A12ml  = XA12*rmrl + YA12*Imrmrl;
                            Bt_val -= bix[k][m] * A12ml;
                        }
                        Bt[k][l] += Bt_val;
                    
                        // C̃_{kl} = -Σ_m [bi]×_{km} A12_{mn} bjx_{nl} 
                        //         - Σ_m [bi]×_{km} B^T12_{ml}
                        //         + Σ_m B12_{km} bjx_{ml}
                        //         + C12_{kl}
                        Scalar C_val = C12kl;
                        for ( int m = 0; m < 3; ++m ) {
                            const Scalar Bt12ml = YB12 * epsr[m][l];
                            const Scalar B12km  = YB12 * epsr[k][m];
                            C_val -= bix[k][m] * Bt12ml;
                            C_val += B12km * bjx[m][l];
                            for ( int n = 0; n < 3; ++n ) {
                                const Scalar rmrn   = rhat[m]*rhat[n];
                                const Scalar Imrmrn = (m==n ? 1.f:0.f) - rmrn;
                                const Scalar A12mn  = XA12*rmrn + YA12*Imrmrn;
                                C_val -= bix[k][m] * A12mn * bjx[n][l];
                            }
                        }
                        C[k][l] += C_val;
                    
                        // ---- Self block (11 scalars, bjx->bix, B^T sign flip) ----
                        const Scalar A11kl  = XA11*rkrl + YA11*Imrkrl;
                        const Scalar Bt11kl = YB11 * epsr[k][l];
                        const Scalar B11kl  = YB11 * epsr[l][k];
                        const Scalar C11kl  = XC11*rkrl + YC11*Imrkrl;

                        // Ã_s_{kl} = A11_{kl}
                        Ã_s[k][l] += A11kl;
                    
                        // B̃^T_s_{kl} = Σ_m A11_{km} bix[m][l] + B^T11_{kl}   (JO sign flip)
                        Scalar sBtT_val = Bt11kl;
                        for ( int m = 0; m < 3; ++m ) {
                            const Scalar rkrm   = rhat[k]*rhat[m];
                            const Scalar Imrkrm = (k==m ? 1.0:0.0) - rkrm;
                            const Scalar A11km  = XA11*rkrm + YA11*Imrkrm;
                            sBtT_val += A11km * bix[m][l];
                        }
                        BtT_s[k][l] += sBtT_val;
                    
                        // B̃_s_{kl} = -Σ_m bix[k][m] A11_{ml} + B11_{kl}
                        Scalar sBt_val = B11kl;
                        for ( int m = 0; m < 3; ++m ) {
                            const Scalar rmrl   = rhat[m]*rhat[l];
                            const Scalar Imrmrl = (m==l ? 1.0:0.0) - rmrl;
                            const Scalar A11ml  = XA11*rmrl + YA11*Imrmrl;
                            sBt_val -= bix[k][m] * A11ml;
                        }
                        Bt_s[k][l] += sBt_val;
                    
                        // C̃_s_{kl} = -[bi]×A11[bi]× - [bi]×B^T11 + B11[bi]× + C11
                        //           = C11 - [bi]×_{km} A11_{mn} bix_{nl}
                        //                 - [bi]×_{km} B^T11_{ml}       (note: -[bi]×B^T not +)
                        //                 + B11_{km} bix_{ml}
                        Scalar sC_val = C11kl;
                        for ( int m = 0; m < 3; ++m ) {
                            const Scalar Bt11ml = YB11 * epsr[m][l];
                            const Scalar B11km  = YB11 * epsr[m][k];
                            sC_val -= bix[k][m] * Bt11ml;
                            sC_val += B11km * bix[m][l];
                            for ( int n = 0; n < 3; ++n ) {
                                const Scalar rmrn   = rhat[m]*rhat[n];
                                const Scalar Imrmrn = (m==n ? 1.f:0.f) - rmrn;
                                const Scalar A11mn  = XA11*rmrn + YA11*Imrmrn;
                                sC_val -= bix[k][m] * A11mn * bix[n][l];
                            }
                        }
                        C_s[k][l] += sC_val;
                    
                      } // l
                    } // k

              } // sb  (inner sphere-j loop)
        } // sa  (inner sphere-i loop)

        // -------------------------------------------------------------------
        //  Write cross-block COO entries for β ≠ α
        // -------------------------------------------------------------------
        for ( int ii = 0; ii < 3; ++ii ) {
            const int row = 6 * (int)tidx + ii;

            for ( int jj = 0; jj < 3; ++jj ) {

                const int col_neigh = 6 * (int)curr_neigh_agg + jj;
		const int oind_neigh = (int)offset_particle + 6 * (int)nneigh * ii + 6 * (int)neigh_idx + jj;

                d_L_RowInd[ oind_neigh ]     = row;   // Ã
                d_L_ColInd[ oind_neigh ]     = col_neigh;
                d_L_Val[    oind_neigh ]     = Ã[ii][jj];

                d_L_RowInd[ oind_neigh + 3 ]   = row;   // B̃^T
                d_L_ColInd[ oind_neigh + 3 ]   = col_neigh + 3;
                d_L_Val[    oind_neigh + 3 ]   = BtT[ii][jj];

                d_L_RowInd[ (int)offset_BC + oind_neigh ]   = row + 3;   // B̃
                d_L_ColInd[ (int)offset_BC + oind_neigh ]   = col_neigh;
                d_L_Val[    (int)offset_BC + oind_neigh ]   = Bt[ii][jj];

                d_L_RowInd[ (int)offset_BC + oind_neigh + 3 ] = row + 3;   // C̃
                d_L_ColInd[ (int)offset_BC + oind_neigh + 3 ] = col_neigh + 3;
                d_L_Val[    (int)offset_BC + oind_neigh + 3 ] = C[ii][jj];

		const int col_self = 6 * (int)tidx + jj;
		const int oind_self = (int)offset_particle + 6 * (int)nneigh * ii + 6 * (int)nneigh_less + jj;

		if ( flag < 1 ){
                    d_L_RowInd[ oind_self ]     = row;   // Ã self
                    d_L_ColInd[ oind_self ]     = col_self;

		    d_L_RowInd[ oind_self + 3 ]   = row;   // B̃^T self
		    d_L_ColInd[ oind_self + 3 ]   = col_self + 3;

		    d_L_RowInd[ (int)offset_BC + oind_self ]   = row + 3;   // B̃ self
		    d_L_ColInd[ (int)offset_BC + oind_self ]   = col_self;

		    d_L_RowInd[ (int)offset_BC + oind_self + 3 ] = row + 3;   // C̃ self
		    d_L_ColInd[ (int)offset_BC + oind_self + 3 ] = col_self + 3;
		}

                d_L_Val[    oind_self ]    += Ã_s[ii][jj]; // Ã self
		d_L_Val[    oind_self + 3 ]  += BtT_s[ii][jj]; // B̃^T self
		d_L_Val[    (int)offset_BC + oind_self ]  += Bt_s[ii][jj]; // B̃ self
		d_L_Val[    (int)offset_BC + oind_self + 3 ] += C_s[ii][jj]; // C̃ self

            } // jj
        } // ii

        flag++;

    } // neigh_idx  (aggregate neighbor loop)

    // -----------------------------------------------------------------------
    //  No-neighbor fallback: write zero placeholder diagonals so that
    //  AddIdentity_kernel has slots to add into.  Matches sphere code exactly.
    // -----------------------------------------------------------------------
    if ( flag == 0 ) {
        for ( unsigned int ii = 0; ii < 3; ++ii ) {
            const int row = 6 * (int)tidx + ii;

            d_L_RowInd[ offset_particle + ii ]             = row;
            d_L_ColInd[ offset_particle + ii ]             = row;
            d_L_Val[    offset_particle + ii ]             = 0.0;

            d_L_RowInd[ offset_BC + offset_particle + ii ] = row + 3;
            d_L_ColInd[ offset_BC + offset_particle + ii ] = row + 3;
            d_L_Val[    offset_BC + offset_particle + ii ] = 0.0;
        }
    }

}// Precondition_RFU_Agg_kernel



/*!
	Build sparse representation for RFU preconditioner. Wrap the functions to build RFU
	in COO format then convert to CSR.

	COO  (row, column, value)
	CSR  (value, column_index, row_index), where the row_index is more of a count. 

	d_pos			(input)  particle positions
	d_group_members		(input)  array of particle indices
	group_size		(input)  Number of particles
	box			(input)  simulation box information
	d_nneigh		(input)  list of number of neighbors for each particle
	d_nneigh_less		(input)  number of neighbors with index less than current particle
	d_nlist			(input)  neighborlist array
	d_headlist		(input)  indices into the neighborlist for each particle
	d_NEPP			(input)  Number of non-zero entries per particle
	d_offset		(input)  current particle's offsets into the output arrays
	d_ResTable_dist		(input)  distances for which the resistance function has been tabulated
	d_ResTable_vals		(input)  tabulated values of the resistance tensor
	table_dr		(input)  table discretization (in log units)
	nnz			(input)  number of non-zero elements in the sparse RFU
	d_L_RowInd		(output) COO row indices
	d_L_RowPtr		(output) CSR row pointer
	d_L_ColInd		(output) COO/CSR col indices
	d_L_Val			(output) COO/CSR vals
	spHandle		(input)  opaque handle for cuSPARSE operations
	rp			(input)  preconditioner cutoff radius
	grid			(input)  Grid for CUDA kernel launch
	threads			(input)  Threads for CUDA kernel launch

*/
void Precondition_Build(
			int *d_HasNeigh,
			Scalar4 *d_pos,
			unsigned int *d_rtag,
			Scalar3 *d_rel_pos,
			int N_agg,
			int nb,
			BoxDim box,
			const unsigned int *d_nneigh_pruned, 
			unsigned int *d_nneigh_less, 
			unsigned int *d_nlist_pruned, 
			const unsigned int *d_headlist_pruned, 
			unsigned int *d_NEPP,
			unsigned int *d_offset, 
			const Scalar *d_ResTable_dist,
			const Scalar *d_ResTable_vals,
			const Scalar table_min,
			const Scalar table_dr,
			int &nnz,
			int   *d_L_RowInd,
			int   *d_L_RowPtr,
			int   *d_L_ColInd,
			Scalar *d_L_Val,
			cusparseHandle_t spHandle,
			const Scalar rp,
			dim3 grid,
			dim3 threads
			){
	
	// Zero the value arrays (Have to zero because the diagonal terms need to be
	// added, and we need to remove any data left over from previous calculations)
	Precondition_ZeroVector_kernel<<<grid, threads>>>( d_L_Val, nnz, N_agg );

	// Build the lubrication tensors
	Precondition_RFU_kernel<<<grid,threads>>>(
							N_agg,
						        nb,	
							d_rtag,
							d_rel_pos,
							d_nneigh_pruned, 
							d_nneigh_less, 
							d_nlist_pruned, 
							d_headlist_pruned, 
							d_NEPP,
							d_offset, 
							d_pos,
							box,
							d_ResTable_dist,
							d_ResTable_vals,
							table_min,
							table_dr,
							d_L_RowInd,   //output
							d_L_ColInd,   //output
							d_L_Val,      //output
							rp
							);
		
	// Convert from COO to CSR (need constant pointers for Row Indices)
	cusparseXcoo2csr( spHandle, d_L_RowInd, nnz, 6*N_agg, d_L_RowPtr, CUSPARSE_INDEX_BASE_ZERO );

	//printCSRMatrix(6*N_agg, nnz, d_L_RowPtr, d_L_ColInd, d_L_Val);//debug

	Precondition_AddGeometricReg_kernel<<<grid,threads>>>(
								N_agg,
								nb,
								d_rtag,
								d_rel_pos,
								d_HasNeigh,
								d_L_RowPtr,
								d_L_ColInd,
								d_L_Val,
								1.0
								);

}

/*!
	Do Reverse-Cuthill-Mckee Reordering of the near-field lubrication tensor preconditioner

	group_size		(input)		number of particles
	d_prcm			(output)	RCM permutation vector
	nnz			(input)		number of non-zero elements in RFU
	d_headlist_pruned	(input)		headlist into pruned neighborlist array
	d_nlist_pruned		(input)		pruned neighborlist array
	d_nneigh_pruned		(input)		pruned number of neighbors
	d_L_RowPtr		(input/output)	CSR row pointer for RFU (before/after reordering)
	d_L_ColInd		(input/output)  CSR column indices for RFU (before/after reordering)
	d_L_Val			(input/output)	CSR values for RFU (before/after reordering)
	soHandle		(input) 	opaque handle for cuSOLVER
	spHandle		(input)		opaque handle for cuSPARSE
	descr_R			(input)		cuSPARSE matrix description of RFU
	d_Scratch3		(input)		Scratch space for re-ordering
	grid			(input)		grid for CUDA kernel launch
	threads			(input)		threads for CUDA kernel launch
	d_scratch		(input)		workspace for index projection
	d_map			(input)		workspace for reorder mapping

*/
void Precondition_Reorder(
				int group_size, 
				int *d_prcm,
				int &nnz,
				unsigned int *d_headlist_pruned,
				unsigned int *d_nlist_pruned,
				unsigned int *d_nneigh_pruned,
				int   *d_L_RowPtr,
				int   *d_L_ColInd,
				Scalar *d_L_Val,
				cusolverSpHandle_t soHandle,
				cusparseHandle_t spHandle,
				cusparseMatDescr_t descr_R,
				Scalar *d_Scratch3,
				dim3 grid,
				dim3 threads,
				int *d_scratch,
				int *d_map
				){	
	
	// Length of nneigh
	int NeighTotal;
	cudaMemcpy( &NeighTotal, &d_headlist_pruned[group_size], sizeof(int), cudaMemcpyDeviceToHost );
	//printf("NeighTotal = %d, N_agg = %d, nnz = %d\n", NeighTotal, group_size, nnz);
		
	// Allocate Host Memory
	int *h_headlist, *h_nlist;
	h_headlist = (int *)malloc( (group_size+1)*sizeof(int) );
	h_nlist    = (int *)malloc( NeighTotal*sizeof(int) );

	int *h_L_RowPtr, *h_L_ColInd;
	h_L_RowPtr = (int *)malloc( (6*group_size+1)*sizeof(int) );
	h_L_ColInd = (int *)malloc( nnz*sizeof(int) );

	int *h_prcm;
	h_prcm = (int *)malloc( (6*group_size)*sizeof(int) );
	
	int *h_map;
	h_map = (int *)malloc( nnz * sizeof(int) );

	// Copy to host
	cudaMemcpy( h_L_RowPtr, d_L_RowPtr, (6*group_size+1)*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_L_ColInd, d_L_ColInd, nnz*sizeof(int), cudaMemcpyDeviceToHost );

	cudaMemcpy( h_headlist, d_headlist_pruned, (group_size+1)*sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_nlist, d_nlist_pruned, NeighTotal*sizeof(int), cudaMemcpyDeviceToHost );

        //Debug
        // Rough condition number from diagonal ratio:
//	Scalar *h_val;
//	h_val = (Scalar *)malloc( (nnz)*sizeof(Scalar) );
//	cudaMemcpy( h_val, d_L_Val, nnz*sizeof(Scalar), cudaMemcpyDeviceToHost );
//        Scalar diag_max = 0.f, diag_min = 1e30f;
//        for (int row = 0; row < 6*group_size; ++row) {
//            for (int jj = h_L_RowPtr[row]; jj < h_L_RowPtr[row+1]; ++jj) {
//                if (h_L_ColInd[jj] == row) {
//                    diag_max = max(diag_max, fabsf(h_val[jj]));
//                    diag_min = min(diag_min, fabsf(h_val[jj]));
//                }
//            }
//        }
//        printf("diag ratio (rough condition estimate) = %.4e\n", diag_max/diag_min);
//	free(h_val);
//
	
	// Use John Burkardt's code for RCM reordering. Alternative routines are given in
	// cuSOLVER and BOOST libraries, but both those implementations are very slow, and
	// exhibit superlinear scaling of the computational cost with number of particles.  
	//
	// Burkardt code expects 1-based lists for the neighbor adjacency.
	//
	// Burkardt code gives 1-based indexing for prcm, so fix that too.
	//
	// TODO: Put the 1-based indexing fix into ExpandPRCM_kernel
	for ( int ii = 0; ii < NeighTotal; ++ii ){
		h_nlist[ ii ] += 1;
	}
	for ( int ii = 0; ii < group_size+1; ++ii ){
		h_headlist[ ii ] += 1;
	}

	//cusolverSpXcsrsymamdHost(
	//			soHandle, 
	//			6*group_size,
	//			nnz,
	//			descr_R,
	//			h_L_RowPtr,
	//			h_L_ColInd,
	//			h_prcm
	//			);
	genrcm( group_size, NeighTotal, h_headlist, h_nlist, h_prcm );

	for( int ii = 0; ii < group_size; ++ii ){
		h_prcm[ ii ] -= 1;
	}

	// Expand the re-ordering from particle-based to 6N index-based
	cudaMemcpy( d_scratch, h_prcm, group_size*sizeof(int), cudaMemcpyHostToDevice );
	Precondition_ExpandPRCM_kernel<<< grid, threads >>>(
							d_prcm,
							d_scratch,
							group_size
							);
	cudaMemcpy( h_prcm, d_prcm, 6*group_size*sizeof(int), cudaMemcpyDeviceToHost );

	// Find the Buffer Size required to apply the permutation
	size_t pBufferSizePermute = 0;
	cusolverSpXcsrperm_bufferSizeHost(
					soHandle,
					6*group_size,
					6*group_size,
					nnz,
					descr_R,
					h_L_RowPtr,
					h_L_ColInd,
					h_prcm,
					h_prcm,
					&pBufferSizePermute
					);

	// Allocate buffer for reordering
	void *pBuffer;
	pBuffer = (void *)malloc( pBufferSizePermute );

	// Create a Map for the permutation
	int block_size = 256;
	dim3 val_grid( int(nnz/block_size) + 1, 1, 1 );
	dim3 val_threads(block_size, 1, 1);
	
	Precondition_InitializeMap_kernel<<< val_grid, val_threads >>>( d_map, nnz );
	cudaMemcpy( h_map, d_map, nnz*sizeof(int), cudaMemcpyDeviceToHost );

	// Do the permutation
	cusolverSpXcsrpermHost(
				soHandle,
				6*group_size,
				6*group_size,
				nnz,
				descr_R,
				h_L_RowPtr,
				h_L_ColInd,
				h_prcm,
				h_prcm,
				h_map,
				pBuffer
				);

	// Copy result to the GPU
	cudaMemcpy( d_L_RowPtr, h_L_RowPtr, (6*group_size+1)*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_L_ColInd, h_L_ColInd, nnz*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_map, h_map, nnz*sizeof(int), cudaMemcpyHostToDevice );
			
	// Apply the map to the values as well
	Precondition_Map_kernel<<< val_grid, val_threads >>>( d_Scratch3, d_L_Val, d_map, nnz );
	cudaMemcpy( d_L_Val, d_Scratch3, nnz*sizeof(Scalar), cudaMemcpyDeviceToDevice );

	//
	// Clean Up
	//
	free( h_headlist );
	free( h_nlist );

        free( h_map );
        free( pBuffer );

	free( h_L_RowPtr );
	free( h_L_ColInd );
	free( h_prcm );
}

/*!
	Do the incomplete Cholesky decomposition and set up the cuSPARSE
	matrix descriptions for the lubrication preconditioner

	group_size	(input)		Number of particles
	nnz		(input)		Number of nonzero elements 
	d_L_RowPtr	(input/output)	CSR row pointer to RFU / lower Cholesky factor
	d_L_ColInd	(input/output)	CSR col indices to RFU / lower Cholesky factor
	d_L_Val		(input/output)	CSR values for RFU / lower Cholesky factor
	spHandle	(input)		opaque handle for cuSPARSE operations
	spStatus	(input)		status output for cuSPARSE operations
	descr_R		(input)		cuSPARSE matrix description of RFU
	descr_L		(input)		cuSPARSE matrix description for lower Cholesky factor
	info_R		(input)		cuSPARSE info for RFU
	info_L		(input)		cuSPARSE info for lower Cholesky factor
	info_Lt		(input)		cuSPARSE info for upper Cholesky factor
	trans_L		(input)		cuSPARSE transpose operation for lower Cholesky factor
	trans_Lt	(input)		cuSPARSE transpose operation for upper Cholesky factor
	policy_R	(input)		cuSPARSE solver policy for R
	policy_L	(input)		cuSPARSE solver policy for lower Cholesky factor
	policy_Lt	(input)		cuSPARSE solver policy for upper Cholesky factor
	pBufferSize	(output)	Buffer size for cuSPARSE operations
	grid		(input)		grid for CUDA kernel launch
	threads		(input)		threads for CUDA kernel launch

*/
void Precondition_IChol(
			int group_size,
			unsigned int nnz,
			int   *d_L_RowPtr,
			int   *d_L_ColInd,
			Scalar *d_L_Val,
			cusparseHandle_t spHandle,
			cusparseStatus_t spStatus,
			cusparseMatDescr_t    descr_R, 
			cusparseMatDescr_t    descr_L, 
			csric02Info_t         info_R,
			csrsv2Info_t          info_L,
			csrsv2Info_t          info_Lt,
			cusparseOperation_t   trans_L,
			cusparseOperation_t   trans_Lt,
			cusparseSolvePolicy_t policy_R, 
			cusparseSolvePolicy_t policy_L,
			cusparseSolvePolicy_t policy_Lt,
			int& pBufferSize,
			dim3 grid,
			dim3 threads,
			Scalar &ichol_relaxer,
			bool &ichol_converged
			){
		
	// 1. Incomplete Cholesky decomposition for the preconditioner needs to 
	//    be performed on ( RFUnf + relaxer*I ), so add Identity to the matrix
	Precondition_AddIdentity_kernel<<<grid,threads>>>(
							d_L_Val,
							d_L_RowPtr,
							d_L_ColInd, 
							group_size,
							ichol_relaxer
							);
		
        // 2. Get buffer memory requirements and allocate buffer
	int pBufferSize_R = 0;  // Buffer size required for calculations on R
	int pBufferSize_L = 0;  // Buffer size required for calculations on L
	int pBufferSize_Lt = 0; // Buffer size required for calculations on L^T

	int numel = 6*group_size;

	cusparseDcsric02_bufferSize(
					spHandle,
					numel,
					nnz,
					descr_R,
					d_L_Val,
					d_L_RowPtr,
					d_L_ColInd,
					info_R,
					&pBufferSize_R
					);
 
	cusparseDcsrsv2_bufferSize(
					spHandle,
					trans_L,
					numel,
					nnz,
					descr_L,
					d_L_Val,
					d_L_RowPtr,
					d_L_ColInd,
					info_L,
					&pBufferSize_L
					);
	
	cusparseDcsrsv2_bufferSize(
					spHandle,
					trans_Lt,
					numel,
					nnz,
					descr_L,
					d_L_Val,
					d_L_RowPtr,
					d_L_ColInd,
					info_Lt,
					&pBufferSize_Lt
					);
	
    	pBufferSize = max( pBufferSize_R, max(pBufferSize_L, pBufferSize_Lt) );
	//std::cout << "pBufferSize = " << pBufferSize << std::endl; //zhoge: GPUdebug
	void *pBuffer;
	cudaMalloc((void**)&pBuffer, pBufferSize );

	// 3. Parameters and initialization needed for cuSPARSE procedures
	int numerical_zero;  // Checks for zero pivots in matrix decomposition
	int structural_zero;

	// 4. Pre-solve analysis
	cusparseDcsric02_analysis(
				spHandle,
				numel,
				nnz,
				descr_R,
				d_L_Val,
				d_L_RowPtr,
				d_L_ColInd,
				info_R,
				policy_R,
				pBuffer
				);

	cusparseDcsrsv2_analysis(
				spHandle,
				trans_L,
				numel,
				nnz,
				descr_L,
				d_L_Val,
				d_L_RowPtr,
				d_L_ColInd,
				info_L,
				policy_L,
				pBuffer
				);

	cusparseDcsrsv2_analysis(
				spHandle,
				trans_Lt,
				numel,
				nnz,
				descr_L,
				d_L_Val,
				d_L_RowPtr,
				d_L_ColInd,
				info_Lt,
				policy_Lt,
				pBuffer
				);

	// Check for zero pivot in the incomplete Cholesky decomposition
	spStatus = cusparseXcsric02_zeroPivot(spHandle, info_R, &structural_zero);
	if ( CUSPARSE_STATUS_ZERO_PIVOT == spStatus ){
		printf("R(%d,%d) is missing \n", structural_zero, structural_zero);
		exit(1);
	}
		
        // 5. Perform incomplete Cholesky decomposition, (RFUnf + relaxer*I) = L * L'
	//
	spStatus = cusparseDcsric02(
					spHandle,
					numel,
					nnz,
					descr_R,
					d_L_Val,      //input/output
					d_L_RowPtr,
					d_L_ColInd,
					info_R,
					policy_R,
					pBuffer
					);
			
	if ( spStatus != CUSPARSE_STATUS_SUCCESS) {
		printf("    Incomplete Cholesky Failed. Quitting.\n");
		Debug_StatusCheck_cuSparse( spStatus, "Ichol decomposition" );
		exit(1);
	}
	
	// Check for numerical zero (loss of positive-definite)	
	spStatus = cusparseXcsric02_zeroPivot(spHandle, info_R, &numerical_zero);
	if ( CUSPARSE_STATUS_ZERO_PIVOT == spStatus ){
	  //originally commented (zhoge: GPUdebug)
	  //printf("L(%d,%d) is zero \n", numerical_zero, numerical_zero);
	  //exit(1);

		// Set convergence flag to false and increase the relaxation
		// parameter for the next time
		ichol_converged = false;
		ichol_relaxer *= 2.0f;
		
		// Return
		cudaFree(pBuffer); //zhoge: GPUdebug
		return;
	}
	
	// Set converged to true if we get this far
	ichol_converged = true;

	// Zero any entries above the main diagonal (Have to do this because,
	// at least as of the cuda-8.0 toolkit, even specifying 
	// MATRIX_FILL_MODE=LOWER doesn't prevent the CUDA functions from using
	// the full matrix in the various solves and mutiplications. 
	Precondition_ZeroUpperTriangle_kernel<<<grid,threads>>>( 
								d_L_RowPtr,
								d_L_ColInd,
								d_L_Val,
								group_size
								);

	// Clean up
	cudaFree( pBuffer );
}

/*
	Wrapper for all the functions required to build the preconditioner, in the proper order

	zhoge: It mainly does S = L * L^T, where S = P * (R_FU^nf + relaxer*I) * P^T, 
               R_FU^nf is the pruned near-field RFU,
               relaxer is in powers of 2 (starting from 1), 
               P is the RCM permutation matrix (P^T its inverse),
               and L is a lower incomplete Cholesky factor (L^T its upper factor).

	d_pos			(input)		particle positions
	d_group_members		(input)		indices for particles in the integration group
	group_size		(input)		number of particles
	box			(input)		periodic box information
	ker_data		(input)		structure containing CUDA kernel information
	res_data		(input/output)	structure containing resistance and preconditioner information

*/
void Precondition_Wrap(
			Scalar4 *d_pos,
			unsigned int *d_rtag,
			const BoxDim& box,
			KernelData *ker_data,
			ResistanceData *res_data,
			WorkData *work_data
			){
	
	// Get kernel information
	dim3 grid = ker_data->rigid_grid;
	dim3 threads = ker_data->rigid_threads;

	int N_agg = res_data->N_rigid;
	int nb = res_data->m_nb;

	//print_neighbor_list_kernel<<<ker_data->particle_grid, ker_data->particle_threads>>>(res_data->nlist,res_data->headlist,res_data->nneigh,N_agg*nb);

	// Check whether particle has neighbors within the lubrication cutoff
	Precondition_HasNeigh_kernel<<< grid, threads >>>( 	
								res_data->HasNeigh,  //output
								N_agg,
								nb,
								d_pos,
								box,
								d_rtag,
								res_data->body_tag,
								res_data->nneigh, 
								res_data->nlist, 
								res_data->headlist,
								res_data->rlub
								);

	// Get the pruned neighbor list (within rp instead of rlub)
	Precondition_PruneNeighborList(
					d_pos,
					d_rtag,
					box,
					res_data,  //output
					ker_data
					);
	
	// Pre-process the arrays for the preconditioner (count certain non-zeros)
	Precondition_PreProcess(
				N_agg, 
				res_data->nnz,
				res_data->nneigh_pruned, 
				res_data->nlist_pruned, 
				res_data->headlist_pruned, 
				res_data->NEPP,            //output
				res_data->offset,	   //output
				ker_data->rigid_grid,
				ker_data->rigid_threads
				);
		
	// Build the approximate lubrication tensor, R_FU^nf (output in COO and CSR formats)
	Precondition_Build(
				res_data->HasNeigh,
				d_pos,
				d_rtag,
				res_data->rel_pos,
				N_agg,
				nb,
				box,
				res_data->nneigh_pruned, 
				res_data->nneigh_less, 
				res_data->nlist_pruned, 
				res_data->headlist_pruned, 
				res_data->NEPP,
				res_data->offset, 
				res_data->table_dist,
				res_data->table_vals,
				res_data->table_min,
				res_data->table_dr,
				res_data->nnz,
				res_data->L_RowInd,   //output
				res_data->L_RowPtr,   //output
				res_data->L_ColInd,   //output
				res_data->L_Val,      //output
				res_data->spHandle,
				res_data->rp,
				ker_data->rigid_grid,
				ker_data->rigid_threads
				);
	//printf("Neighbor list is\n");
	//print_neighbor_list_kernel<<<grid, threads>>>(res_data->nlist_pruned,res_data->headlist_pruned,res_data->nneigh_pruned,N_agg);
	//printf("build matrix is\n");
	//printCSRMatrix(6*N_agg, res_data->nnz, res_data->L_RowPtr, res_data->L_ColInd, res_data->L_Val);
	// Re-order the lubrication tensor (R_FU^nf) using RCM (using an implementation in rcm.cpp)
	// zhoge: Should result P * (R_FU^nf) * P^T
	Scalar *d_Scratch3 = (res_data->Scratch3);
	Precondition_Reorder(
				N_agg, 
				res_data->prcm,                   //output (the permutation)
				res_data->nnz,
				res_data->headlist_pruned,
				res_data->nlist_pruned,
				res_data->nneigh_pruned,
				res_data->L_RowPtr,               //input/output
				res_data->L_ColInd,		  //input/output
				res_data->L_Val,		  //input/output
				res_data->soHandle,
				res_data->spHandle,
				res_data->descr_R,
				d_Scratch3,
				ker_data->rigid_grid,
				ker_data->rigid_threads,
				work_data->precond_scratch,
				work_data->precond_map
				);

	//printf("reordered matrix is\n");
	//printCSRMatrix(6*N_agg, res_data->nnz, res_data->L_RowPtr, res_data->L_ColInd, res_data->L_Val);

	
	// Get the inverse square root of the diagonal elements (related to near-field Brownian calculations)
	Precondition_GetDiags_kernel<<< grid, threads >>>(
								N_agg, 
								res_data->Diag,     //output
								res_data->L_RowPtr, //input
								res_data->L_ColInd, //input
								res_data->L_Val	    //input
								);
	/* //zhoge: redundent (done in IChol below)
	// Add far-field contribution to the diagonal, i.e. S = R_FU^nf + ichol_relaxer*(1 or 4/3)
	Precondition_AddIdentity_kernel<<<grid,threads>>>(
							  res_data->L_Val,     //input/output
							  res_data->L_RowPtr,  //input
							  res_data->L_ColInd,  //input
							  group_size,	       //input
							  1.0                  //input: relaxation factor
							  );
	
	// Check if there are zero diagonals (in Helper_Debug.cu)
	Debug_CSRzeroDiag( res_data->L_RowPtr, res_data->L_ColInd, res_data->L_Val, group_size, res_data->nnz );
	*/

	// Backup storage for the elements
	Scalar *d_backup = (work_data->precond_backup);
	Scalar *d_values = (res_data->L_Val);
	cudaMemcpy( d_backup, d_values, (res_data->nnz)*sizeof(Scalar), cudaMemcpyDeviceToDevice );	

	// Set convergence flag false to start so that we try at least once. Then, do the IChol
	// factorization, adding along the diagonal as needed to ensure convergence.

	res_data->ichol_relaxer = 1.0;
	(res_data->ichol_converged) = false;
	//int idebug = 0;
	while ( !(res_data->ichol_converged) ){

	  // Copy original values (res_data->L_Val) pointed to by d_values
	  cudaMemcpy( d_values, d_backup, (res_data->nnz)*sizeof(Scalar), cudaMemcpyDeviceToDevice );

	  // Do the incomplete Cholesky decomposition (L * L^T)
	  // output replaces the input
	  Precondition_IChol(
			     N_agg,
			     res_data->nnz,
			     res_data->L_RowPtr,      //input/output
			     res_data->L_ColInd,      //input/output
			     res_data->L_Val,	      //input/output
			     res_data->spHandle,
			     res_data->spStatus,
			     res_data->descr_R, 
			     res_data->descr_L, 
			     res_data->info_R,
			     res_data->info_L,
			     res_data->info_Lt,
			     res_data->trans_L,
			     res_data->trans_Lt,
			     res_data->policy_R, 
			     res_data->policy_L,
			     res_data->policy_Lt,
			     res_data->pBufferSize,    //output
			     ker_data->rigid_grid,
			     ker_data->rigid_threads,
			     res_data->ichol_relaxer,  //relaxation factor: can be modified; reset to 1.0 periodically, see Stokes.cc
			     res_data->ichol_converged
			     );
	  //idebug++; //zhoge
	}
	//if (idebug > 1)
	//  std::cout << "IChol iterations " << idebug << std::endl; //zhoge: GPUdebug

	//printf("ichol relaxer is %f\n",res_data->ichol_relaxer);
	//printCSRMatrix(6*N_agg, res_data->nnz, res_data->L_RowPtr, res_data->L_ColInd, res_data->L_Val);

	// Cleanup
	d_Scratch3 = NULL;
	d_backup = NULL;
	d_values = NULL;
	
}

/*
	Preconditioned RFU multiply for the Brownian calculation. 

		y = L^(-1) * D^(-1) * P * ( R_FU^nf + I_nn ) * P^(T) * D^(-T) * L^(-T) * x

		L   = Lower Cholesky factor of ( \tilde R_FU^nf + I )
		D   = Modified diagonal elements of R_FU
		P   = RCM re-ordering
		I_nn = modified identity tensor (non-zero only if no neighbor)

	zhoge: The order of D and P is inconsistent with the FSD paper, 
               but it is correct because D was obtained from the permutated RFU.

	!!! CAN work with in-place solve (i.e. pointers d_x = d_y)

	d_y			(output) product of preconditioned matrix-vector multiply
	d_x			(input)  vector to multiply by preconditioned matrix
	d_pos			(input)  particle positions
	d_group_members		(input)  list of particle indices within the integration group
	group_size		(input)  number of particles
	box			(input)  periodic box information
	ker_data		(input)  structure containing information for CUDA kernel launches
	res_data		(input)  structure containing information for resistance calculations

*/
void Precondition_Brownian_RFUmultiply(	
					Scalar *d_y, // output
					Scalar *d_x, // input
					const Scalar4 *d_pos,
					unsigned int *d_group_members,
					const int group_size, 
			      		const BoxDim box,
					void *pBuffer,
					KernelData *ker_data,
					ResistanceData *res_data
					){
	
	// Kernel data
	dim3 grid    = (ker_data->particle_grid);
	dim3 threads = (ker_data->particle_threads);

	//// Pointer to scratch array (size 6N, same as d_x and d_y)
	//Scalar *d_z = (res_data->Scratch1);
	//
	//// Number of elements of the arrays
	//int numel = 6 * group_size;
	//
	//// Variable required for Axpy
	//Scalar spAlpha = 1.0;
	//
	////
	////// First incomplete Cholesky Solve: solve L'*y = x
	//cusparseDcsrsv2_solve(
	//			res_data->spHandle, 
	//			res_data->trans_Lt, 
	//			numel, 
	//			res_data->nnz, 
	//			&spAlpha, 
	//			res_data->descr_L,
	//   			res_data->L_Val, 
	//			res_data->L_RowPtr, 
	//			res_data->L_ColInd, 
	//			res_data->info_Lt,
	//   			d_x, // input
	//			d_y, // output
	//			res_data->policy_Lt, 
	//			pBuffer
	//			);
	//
	//// Diagonal multiplication (zhoge: Actually dividing the diagonal elements by their square roots)
	//Precondition_DiagMult_kernel<<< grid, threads >>>(
	//							d_z, // output
	//							d_y, // input
	//							group_size, 
	//							res_data->Diag,  //input
	//							1
	//							);
	//
	//// Permute the input vector (zhoge: Undo the RCM permutation given -1)
	//Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
	//							d_y, // output
	//							d_z, // input
	//							res_data->prcm,
	//							group_size,
	//							-1
	//							);
	//
	//// RFU multiplication and addition of Inn
	////
	//// d_z = RFU * d_y
	//Lubrication_RFU_kernel<<< grid, threads >>>(
	//						d_z, // output (intermediate)
	//						d_y, // input
	//						d_pos,
	//						d_group_members,
	//						group_size, 
	//						box,
	//						res_data->nneigh, 
	//						res_data->nlist, 
	//						res_data->headlist, 
	//						res_data->table_dist,
	//						res_data->table_vals,
	//						res_data->table_min,
	//						res_data->table_dr,
	//						res_data->rlub
	//						);
	//// d_z += Inn * d_y
	//Precondition_Inn_kernel<<< grid, threads >>>(
	//						d_z, // input/output (overwritten)
	//						d_y, // input
	//						res_data->HasNeigh,
	//						group_size
	//						);
	//
	//// Permute the output vector (zhoge: Apply the RCM given 1)
	//Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
	//							d_y, // output
	//							d_z, // input
	//							res_data->prcm,
	//							group_size,
	//							1
	//							);
	//
	//// Diagonal multiplication (zhoge: Divide again, now the diagonals are 1 if they were between 0 and 1)
	//Precondition_DiagMult_kernel<<< grid, threads >>>(
	//							d_z, // output
	//							d_y, // input
	//							group_size, 
	//							res_data->Diag,
	//							1
	//							);
	//
	//// Second incomplete Cholesky solve: solve L*y = x
	//cusparseDcsrsv2_solve(
	//			res_data->spHandle, 
	//			res_data->trans_L, 
	//			numel, 
	//			res_data->nnz, 
	//			&spAlpha, 
	//			res_data->descr_L,
	//			res_data->L_Val, 
	//			res_data->L_RowPtr, 
	//			res_data->L_ColInd, 
	//			res_data->info_L,
	//			d_z, // input 
	//			d_y, // output
	//			res_data->policy_L, 
	//			pBuffer
	//			);
        //// Clean up
        //d_z = NULL;
	
	//debug: effectively turn off the preconditioner
	// d_y = RFU * d_x
	Lubrication_RFU_kernel<<< grid, threads >>>(
						    	d_y, // output
							d_x, // input
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

/*
	Undoes the square root of the pre-conditioner so that the resulting random 
	variable has the correct variance

		x = ( I - Inn ) * P^(T) * D * G

		G   = lower Cholesky factor
		D   = modified diagonal matrix
		P   = RCM re-ordering matrix
		Inn = modified identity tensor

	zhoge: Again, D and P^T are flipped relative to the FSD paper, but it is self-consistent in the code.

	!!! Works in-place

	d_x		(input/output) 	vector to be rescaled and reordered
	group_size	(input)		number of particles
	ker_data	(input)  	structure containing information for CUDA kernel launches
	res_data	(input)  	structure containing information for resistance calculations

*/
void Precondition_Brownian_Undo(	
				Scalar *d_x,       // input/output
				int group_size,
				void *pBuffer,
				KernelData *ker_data,
				ResistanceData *res_data
				){

	// Kernel information
	dim3 grid = ker_data->particle_grid;
	dim3 threads = ker_data->particle_threads;

	// Pointer to scratch array
	Scalar *d_z = (res_data->Scratch1);

	// Number of elements in vectors
	int numel = 6*group_size;
	
	// Incomplete Cholesky multiplication
	Scalar spAlpha = 1.0;
	Scalar spBeta = 0.0;

	//Deepak:replaced cusparseScsrmv with generic api cusparseSpMV
	cusparseSpMatDescr_t matA;
	cusparseCreateCsr(&matA, numel, numel, res_data->nnz,
			res_data->L_RowPtr, res_data->L_ColInd, res_data->L_Val,
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

	cusparseDnVecDescr_t vecX, vecY;
	cusparseCreateDnVec(&vecX, numel, d_x, CUDA_R_64F);
	cusparseCreateDnVec(&vecY, numel, d_z, CUDA_R_64F);

	//size_t bufferSize = 0;
	//void *dBuffer = nullptr;
	//cusparseSpMV_bufferSize(res_data->spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
	//			&spAlpha, matA, vecX, &spBeta, vecY,
	//			CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
	//cudaMalloc(&dBuffer, bufferSize);

	cusparseSpMV(res_data->spHandle, res_data->trans_L,
			&spAlpha, matA, vecX, &spBeta, vecY,
			CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, pBuffer);

	// cusparseScsrmv(
	// 			res_data->spHandle,
	// 			res_data->trans_L,
	// 			numel,
	// 			numel,
	// 			res_data->nnz,
	// 			&spAlpha,
	// 			res_data->descr_L,
	// 			res_data->L_Val,
	// 			res_data->L_RowPtr,
	// 			res_data->L_ColInd,
	// 			d_x, // Input
	// 			&spBeta,
	// 			d_z  // Output
	// 			);

	// Diagonal preconditioner (zhoge: Multiply the square root given -1, confusing notation but correct)
	Precondition_DiagMult_kernel<<< grid, threads >>>(
								d_x,  // output
								d_z,  // input
								group_size, 
								res_data->Diag,
								-1 
								);
	
	// Permute the output vector (zhoge: Actually undo the RCM permutation given -1)
	Precondition_ApplyRCM_Vector_kernel<<< grid, threads >>>( 
								 	d_z, // output
								    d_x, // input
									res_data->prcm,
									group_size,
									-1
									);

	// Project out the components for particles with no neighbors
	Precondition_ImInn_kernel<<< grid, threads >>>(
							d_x, // output
							d_z, // input
							res_data->HasNeigh,
							group_size
							);

	// Clean pointers
	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	//cudaFree(dBuffer);
	d_z = NULL;

}




/*
	Apply the preconditioner for the near-field lubrication in the saddle point solve

	Wrapper to perform the solves required of inverting the incomplete
	Cholesky representation of the approximate resistance tensor

		y = ( L * L' ) \ x

	!!! CAN work with in-place solve (i.e. pointers d_x = d_y -- needed for GMRES)
	
	d_y		(output) product of matrix and input vector
	d_x		(input)  input vector for multiplication
	d_Scratch	(input)  scratch space for calculations
	d_prcm		(input)  RCM re-ordering vector
	group_size	(input)  Number of particles
	nnz		(input)  Number of nonzero elements 
	d_L_RowPtr	(input)  CSR row pointer to RFU / lower Cholesky factor
	d_L_ColInd	(input)  CSR col indices to RFU / lower Cholesky factor
	d_L_Val		(input)  CSR values for RFU / lower Cholesky factor
	spHandle	(input)	 opaque handle for cuSPARSE operations
	spStatus	(input)	 status output for cuSPARSE operations
	descr_L		(input)  cuSPARSE matrix description for lower Cholesky factor
	info_L		(input)  cuSPARSE info for lower Cholesky factor
	info_Lt		(input)  cuSPARSE info for upper Cholesky factor
	trans_L		(input)  cuSPARSE transpose operation for lower Cholesky factor
	trans_Lt	(input)  cuSPARSE transpose operation for upper Cholesky factor
	policy_L	(input)  cuSPARSE solver policy for lower Cholesky factor
	policy_Lt	(input)  cuSPARSE solver policy for upper Cholesky factor
	pBufferSize	(input)  Buffer size for cuSPARSE operations
	grid		(input)  grid for CUDA kernel launch
	threads		(input)  threads for CUDA kernel launch



*/
void Precondition_Saddle_RFUmultiply(	
					Scalar *d_y,       // output
					Scalar *d_x,       // input
					Scalar *d_Scratch, // intermediate storage
					const int *d_prcm,
					int group_size,
					unsigned int nnz,
					const int   *d_L_RowPtr,
					const int   *d_L_ColInd,
					const Scalar *d_L_Val,
					cusparseHandle_t spHandle,
					cusparseStatus_t spStatus,
					cusparseMatDescr_t descr_L,
					csrsv2Info_t info_L,
					csrsv2Info_t info_Lt,
					const cusparseOperation_t trans_L,
					const cusparseOperation_t trans_Lt,
					const cusparseSolvePolicy_t policy_L,
					const cusparseSolvePolicy_t policy_Lt,
					void *pBuffer,
					dim3 grid,
					dim3 threads
					){

	// Variable required for Axpy
	Scalar spAlpha = 1.0;

	// Vector length
	int numel = 6 * group_size;
	
	// Permute the input vector
	Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
								d_Scratch, // output
								d_x,       // input
								d_prcm,
								group_size,
								1
								);

	//
	// Incomplete Cholesky solve
	
	// first: solve L*y = x
	cusparseDcsrsv2_solve(
				spHandle, 
				trans_L, 
				numel,
				nnz, 
				&spAlpha, 
				descr_L,
	   			d_L_Val, 
				d_L_RowPtr, 
				d_L_ColInd, 
				info_L,
	   			d_Scratch,  // input 
				d_y,        // output
				policy_L, 
				pBuffer
				);
	
	// second: solve L'*z = y
	cusparseDcsrsv2_solve(
				spHandle, 
				trans_Lt, 
				numel, 
				nnz, 
				&spAlpha, 
				descr_L,
	   			d_L_Val, 
				d_L_RowPtr, 
				d_L_ColInd, 
				info_Lt,
	   			d_y,       // input
				d_Scratch, // output
				policy_Lt, 
				pBuffer
				);

	// Permute the output vector
	Precondition_ApplyRCM_Vector_kernel<<<grid,threads>>>( 
								d_y,       // output
								d_Scratch, // input
								d_prcm,
								group_size,
								-1
								);
	

}
