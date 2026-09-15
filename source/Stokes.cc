// Modified by Andrew Fiore
// Modified by Zhouyang Ge

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

using namespace std;

#include "Stokes.h"
#include "Stokes.cuh"  //zhoge: This includes HOOMDMath.h, which includes cmath

#include "DataStruct.h"

#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <quadmath.h>

#include <cusparse.h>
#include <cusolverSp.h>

/*! \file Stokes.cc
    \brief Contains code for the Stokes class
*/

/*! 
	sysdef		SystemDefinition this method will act on. Must not be NULL.
        group		The group of particles this integration method is to work on
	T		temperature
	seed		Seed for random number generator
	nlist_ewald	neighbor list for Ewald calculation
	xi		Ewald parameter
	m_error		Error tolerance for all calculations
	fileprefix	prefix for output of stresslet data
	period		frequency of output of stresslet data
	ndsr            non-dimensional shear rate (interparticle force)
	kappa           inverse Debye length (electrostatic repulsion)
	k_n             collision spring constant
*/
Stokes::Stokes( std::shared_ptr<SystemDefinition> sysdef,
                std::shared_ptr<ParticleGroup> group,
		std::shared_ptr<ParticleGroup> group_p,
                std::shared_ptr<Variant> kT,
                std::shared_ptr<NeighborList> nlist_ewald,
		Scalar xi, Scalar error, Scalar rcut,
		Scalar F_rep, Scalar F_att, Scalar kappa, Scalar k_n,
		Scalar epsq, Scalar rot_diff, Scalar T_ext, Scalar omega_ext,  
		Scalar F_ext, int period, unsigned int seed,
		Scalar rfd_eps, Scalar delta, std::string fric, Scalar h0, Scalar alpha, Scalar kt, Scalar muf
		)
  : IntegrationMethodTwoStep(sysdef, group),m_group_p(group_p),
    m_T(kT), m_seed(seed), m_nlist_ewald(nlist_ewald), m_xi(xi), m_error(error), m_rcut(rcut),
    m_F_rep(F_rep), m_F_att(F_att), m_kappa(kappa), m_k_n(k_n), m_epsq(epsq), m_rot_diff(rot_diff),
    m_T_ext(T_ext), m_omega_ext(omega_ext), m_F_ext(F_ext), m_period(period), m_rfd_epsilon(rfd_eps),
    delta(delta), m_h0(h0), m_alpha(alpha), m_k_t(kt), m_muf(muf)
    {
	m_exec_conf->msg->notice(5) << "Constructing Stokes" << endl;
	// Hash the User's Seed to make it less likely to be a low positive integer
        m_seed = m_seed * 0x12345677 + 0x12345; m_seed ^= (m_seed >> 16); m_seed *= 0x45679;
	setParams();
	setResistanceTable();
	setSparseMath();
	AllocateWorkSpaces();

	//set tangential friction
        if(fric=="hyd") {
                m_fric= false;
                setFriction();
        }
        else if(fric=="contact") {
                m_fric = true;
                cudaMalloc( (void**)&m_contact_table,    m_pdata->getMaxN() * max_contact * sizeof(ContactSlot) );
                setupContactTable(m_contact_table, m_pdata->getN(), max_contact);
        }
        else m_fric= false;

	//Deepak: moved from step one function
	//Randomize seeds for stochastic calculations
	srand( m_seed);
	m_seed_ff_rs = rand();
	m_seed_ff_ws = rand();
	m_seed_nf = rand();
	m_seed_rfd = rand();

	if(m_period>0) stressfile.open("stress.txt", std::ios_base::out|std::ios::app);
	if (!stressfile.good()) {
		throw std::runtime_error("Error in Stokes::OutputData. Unable to open output file.");
	}
	stressfile.precision(5);

	// only one GPU is supported
	if (!m_exec_conf->isCUDAEnabled())
	{
		throw std::runtime_error("Error initializing Stokes");
	}

	m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "stokes"));
	m_autotuners.push_back(m_tuner);
    }


//! Destructor for the Stokes class
Stokes::~Stokes()
    {
	// Print out
   	m_exec_conf->msg->notice(5) << "Destroying Stokes" << endl;

	// Clean up cuFFT plan
	cufftDestroy(plan);

	// Clean up cuSOLVER handle
	cusolverSpDestroy(soHandle);

	// Clean up cuSPARSE handle and descriptions
	cusparseDestroy(spHandle);

	cusparseDestroyMatDescr(descr_R);
	cusparseDestroyMatDescr(descr_L);

	cusparseDestroyCsric02Info(info_R);
	cusparseDestroyCsrsv2Info(info_L);
	cusparseDestroyCsrsv2Info(info_Lt);
	
	// Clean up cuBLAS handle
	cublasDestroy( blasHandle );
	
	// Free workspace
	FreeWorkSpaces();
	if(m_period>0) stressfile.close();
    }

/*!
	Set the parameters for Spectral Ewald Method
*/
void Stokes::setParams()
{

	// Try two Lanczos iterations to start (number of iterations will adapt as needed)
	m_m_Lanczos_ff = 2;
	m_m_Lanczos_nf = 2;

	// m_rfd_epsilon = 100.0;//pow(m_error,1.0/3.0);

	// At first only need to add identity, then increase if necessary  (used in Precondition.cu for the Cholesky decomposition)
	m_ichol_relaxer = 1.0f;

	// Real space cutoff
	m_ewald_cut = sqrt( - log( m_error ) ) / m_xi;
	
	// Number of grid points
	// int kmax = int( 2.0 * sqrt( - log( m_error ) ) * m_xi ) + 1;
	Scalar kmax = 2.0 * sqrt( - log( m_error ) ) * m_xi;
	
	const BoxDim& box = m_pdata->getBox(); // Only for box not changing with time.
	Scalar3 L = box.getL();

	// Check that rcut is not too large (otherwise interact with images)
	if ( ( m_ewald_cut > L.x/2.0 ) || ( m_ewald_cut > L.y/2.0 ) || ( m_ewald_cut > L.z/2.0 ) ){
		
		Scalar max_cut;
		if ( ( L.x < L.y ) && ( L.x < L.z ) ){
			max_cut = L.x / 2.0;
		}
		else if ( ( L.y < L.x ) && ( L.y < L.z ) ){
			max_cut = L.y / 2.0;
		}
		else if ( ( L.z < L.x ) && ( L.z < L.y ) ){
			max_cut = L.z / 2.0;
		}
		else {
			max_cut = L.x / 2.0;
		}

		Scalar new_xi = sqrt( -log( m_error ) ) / max_cut;

		printf("Real space Ewald cutoff radius is too large! \n");
		printf("    xi = %f \n    rcut = %f \n    box = ( %f %f %f ) \n", m_xi, m_ewald_cut, L.x, L.y, L.z );
		printf("Increase xi to %f or larger to fix. \n", new_xi );
		
		exit(EXIT_FAILURE);

	}
	// initially, at least two points for the smallest wave length (modified to be multiples of 2,3,5 later)
	// m_Nx = int( kmax * L.x / ( 2.0 * 3.1415926536 ) * 2.0 ) + 1; 
	// m_Ny = int( kmax * L.y / ( 2.0 * 3.1415926536 ) * 2.0 ) + 1; 
	// m_Nz = int( kmax * L.z / ( 2.0 * 3.1415926536 ) * 2.0 ) + 1;
	m_Nx = int(std::ceil(2 * kmax * L.x / ( 2.0 * 3.1415926536 )) ) + 1;
	m_Ny = int(std::ceil(2 * kmax * L.y / ( 2.0 * 3.1415926536 )) ) + 1;
	m_Nz = int(std::ceil(2 * kmax * L.z / ( 2.0 * 3.1415926536 )) ) + 1;
	
	// Get list of int values between 8 and 512 that can be written as
	// 	(2^a)*(3^b)*(5^c)
	// Then sort list from low to high and figure out how many entries there are
	std::vector<int> Mlist;
	for ( int ii = 0; ii < 10; ++ii ){
		int pow2 = 1;
		for ( int i = 0; i < ii; ++i ){
			pow2 *= 2;
		}
		for ( int jj = 0; jj < 6; ++jj ){
			int pow3 = 1;
			for ( int j = 0; j < jj; ++j ){
				pow3 *= 3;
			}
			for ( int kk = 0; kk < 4; ++kk ){
				int pow5 = 1;
				for ( int k = 0; k < kk; ++k ){
					pow5 *= 5;
				}
				int Mcurr = pow2 * pow3 * pow5;
				if ( Mcurr >= 8 && Mcurr <= 512 ){
					Mlist.push_back(Mcurr);
				}
			}
		}
	}
	std::sort(Mlist.begin(),Mlist.end());
	const int nmult = static_cast<int>(Mlist.size());

	// Compute the number of grid points in each direction
	//
	// Number of grid points should be a power of 2,3,5 for most efficient FFTs
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nx <= Mlist[ii]){
			 m_Nx = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Ny <= Mlist[ii]){
			m_Ny = Mlist[ii];
			break;
		}
	}
	for ( int ii = 0; ii < nmult; ++ii ){
		if (m_Nz <= Mlist[ii]){
			m_Nz = Mlist[ii];
			break;
		}
	}

	// Maximum number of FFT nodes is limited by available memory
	// Max Number = 512 * 512 * 512 = 134,217,728
	if ( m_Nx * m_Ny * m_Nz > 1024*1024*1024 ){

		printf("Requested Number of Fourier Nodes Exceeds Max Dimension of 512^3\n");
		printf("Mx = %i \n", m_Nx);
		printf("My = %i \n", m_Ny);
		printf("Mz = %i \n", m_Nz);

		exit(EXIT_FAILURE);
	}

        // Maximum eigenvalue of A'*A to scale support, P, for spreading on 
	// deformed grids (Fiore and Swan, J. Chem. Phys., 2018)
	Scalar gamma = m_max_strain;
	Scalar gamma2 = gamma*gamma;
	Scalar lambda = 1.0 + gamma2/2.0 + gamma*sqrt(1.0 + gamma2/4.0);

	// Grid spacing
	m_gridh = L / make_scalar3(m_Nx,m_Ny,m_Nz); 

	// Parameters for the Spectral Ewald Method (Lindbo and Tornberg, J. Comp. Phys., 2011)
	m_gaussm = 1.0;
	while ( erfc( m_gaussm / sqrt(2.0*lambda) ) > m_error ){
	    m_gaussm = m_gaussm + 0.01;
	}
	m_gaussP = int( m_gaussm*m_gaussm / 3.1415926536 )  + 1;

	Scalar w = m_gaussP*m_gridh.x / 2.0;	               // Gaussian width in simulation units
	Scalar xisq  = m_xi * m_xi;
	m_eta = (2.0*w/m_gaussm)*(2.0*w/m_gaussm) * ( xisq );  // Gaussian splitting parameter	

	// Check that the support size isn't somehow larger than the grid
	if ( m_gaussP > std::min( m_Nx, std::min( m_Ny, m_Nz ) ) ){

		printf("Quadrature Support Exceeds Available Grid\n");
		printf("( Mx, My, Mz ) = ( %i, %i, %i ) \n", m_Nx, m_Ny, m_Nz);
		printf("Support Size, P = %i \n", m_gaussP);

		exit(EXIT_FAILURE);
	}

	// Print summary to command line output
	printf("\n");
	printf("\n");
	m_exec_conf->msg->notice(2) << "--- NUFFT Hydrodynamics Statistics ---" << endl;
	m_exec_conf->msg->notice(2) << "Mx: " << m_Nx << endl;
	m_exec_conf->msg->notice(2) << "My: " << m_Ny << endl;
	m_exec_conf->msg->notice(2) << "Mz: " << m_Nz << endl;
	m_exec_conf->msg->notice(2) << "rcut: " << m_ewald_cut << endl;	
	m_exec_conf->msg->notice(2) << "Points per radius (x,y,z): " << m_Nx / L.x << ", " << m_Ny / L.y << ", " << m_Nz / L.z << endl;
	m_exec_conf->msg->notice(2) << "--- Gaussian Spreading Parameters ---"  << endl;
	m_exec_conf->msg->notice(2) << "gauss_m: " << m_gaussm << endl;
    	m_exec_conf->msg->notice(2) << "gauss_P: " << m_gaussP << endl;
	m_exec_conf->msg->notice(2) << "gauss_eta: " << m_eta << endl; 
	m_exec_conf->msg->notice(2) << "gauss_w: " << w << endl; 
	m_exec_conf->msg->notice(2) << "gauss_gridh (x,y,z): " << L.x/m_Nx << ", " << L.y/m_Ny << ", " << L.z/m_Nz << endl;
	printf("\n");
	printf("\n");

	// Create plan for CUFFT on the GPU
	cufftPlan3d(&plan, m_Nx, m_Ny, m_Nz, CUFFT_PLAN_TYPE);

	// Prepare GPUArrays for grid vectors and gridded forces
	GPUArray<Scalar4> n_gridk(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridk.swap(n_gridk);
	GPUArray<CUFFTCOMPLEX> n_gridX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridX.swap(n_gridX);
	GPUArray<CUFFTCOMPLEX> n_gridY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridY.swap(n_gridY);
	GPUArray<CUFFTCOMPLEX> n_gridZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZ.swap(n_gridZ);
	
	GPUArray<CUFFTCOMPLEX> n_gridXX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridXX.swap(n_gridXX);
	GPUArray<CUFFTCOMPLEX> n_gridXY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridXY.swap(n_gridXY);
	GPUArray<CUFFTCOMPLEX> n_gridXZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridXZ.swap(n_gridXZ);
	GPUArray<CUFFTCOMPLEX> n_gridYX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridYX.swap(n_gridYX);
	GPUArray<CUFFTCOMPLEX> n_gridYY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridYY.swap(n_gridYY);
	GPUArray<CUFFTCOMPLEX> n_gridYZ(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridYZ.swap(n_gridYZ);
	GPUArray<CUFFTCOMPLEX> n_gridZX(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZX.swap(n_gridZX);
	GPUArray<CUFFTCOMPLEX> n_gridZY(m_Nx*m_Ny*m_Nz, m_exec_conf);
	m_gridZY.swap(n_gridZY);

	// Get list of reciprocal space vectors, and scaling factor for the wave space calculation at each grid point
	ArrayHandle<Scalar4> h_gridk(m_gridk, access_location::host, access_mode::readwrite);
	for (int i = 0; i < m_Nx; i++) {
	  for (int j = 0; j < m_Ny; j++) {
	    for (int k = 0; k < m_Nz; k++) {

	      // Index into grid vector storage array
	      int idx = i * m_Ny*m_Nz + j * m_Nz + k;

	      // k goes from -N/2 to N/2
	      h_gridk.data[idx].x = 2.0*3.1415926536 * ((i < ( m_Nx + 1 ) / 2) ? i : i - m_Nx) / L.x;
	      h_gridk.data[idx].y = 2.0*3.1415926536 * ((j < ( m_Ny + 1 ) / 2) ? j : j - m_Ny) / L.y;
	      h_gridk.data[idx].z = 2.0*3.1415926536 * ((k < ( m_Nz + 1 ) / 2) ? k : k - m_Nz) / L.z;

	      // k dot k
	      Scalar k2 =
			h_gridk.data[idx].x*h_gridk.data[idx].x +
			h_gridk.data[idx].y*h_gridk.data[idx].y +
			h_gridk.data[idx].z*h_gridk.data[idx].z;

	      // Scaling factor used in wave space sum
	      //
	      // Can't include k=0 term in the Ewald sum
	      if (i == 0 && j == 0 && k == 0) h_gridk.data[idx].w = 0;
	      else
		  		// Have to divide by Nx*Ny*Nz to normalize the FFTs
				h_gridk.data[idx].w = 6.0*3.1415926536 * (1.0 + k2/4.0/xisq) *
				exp( -(1.0-m_eta) * k2/4.0/xisq ) / ( k2 ) / Scalar( m_Nx*m_Ny*m_Nz );
	      
	    }
	  }
	}

	// Store the coefficients for the real space part of Ewald summation
	//
	// Will precompute scaling factors for real space component of summation for a given
	//     discretization to speed up GPU calculations
	//
	// NOTE: Due to the potential sensitivity of the real space functions at smaller xi, the
	//       tabulation will be computed in quadruple precision, then truncated and stored
	//       in single precision
	m_ewald_dr = 0.001; 		           // Distance resolution
	m_ewald_n = int(m_ewald_cut / m_ewald_dr) - 1;  // Number of entries in tabulation

	// Table discretization in quadruple precision
	__float128 dr = 0.00100000000000000000000000000000;

	// Factors needed to compute self contribution
    	Scalar pi12 = 1.77245385091; // square root of pi
	Scalar pi = 3.1415926536;    // pi
    	Scalar aa = 1.0;  	     // radius
	Scalar axi = aa * m_xi;      // a * xi
	Scalar axi2 = axi * axi;     // ( a * xi )^2

	// Compute self contribution
    	m_self.x = (1. + 4.*pi12*axi*erfc(2.*axi) - exp(-4.*axi2))/(4.*pi12*axi*aa);
	m_self.y = ( (-3.*erfc(2.*aa*m_xi)*pow(aa,-3.))/10. - (3.*pow(aa,-6.)*pow(pi,-0.5)*pow(m_xi,-3.))/80. -
		     (9.*pow(aa,-4.)*pow(pi,-0.5)*pow(m_xi,-1.))/40. +
		     (3.*exp(-4.*pow(aa,2.)*pow(m_xi,2.))*pow(aa,-6.)*pow(pi,-0.5)*pow(m_xi,-3.)*
		      (1. + 10.*pow(aa,2.)*pow(m_xi,2.)))/80. );

	// Allocate storage for real space Ewald table
	int nR = m_ewald_n + 1; // number of entries in ewald table
	GPUArray<Scalar4> n_ewaldC1( 2*nR, m_exec_conf); 
	m_ewaldC1.swap(n_ewaldC1);
	ArrayHandle<Scalar4> h_ewaldC1(m_ewaldC1, access_location::host, access_mode::overwrite);

	// Functions are complicated so calculation should be done in quadruple precision, then truncated to single precision
	// in order to ensure accurate evaluation
	__float128 xi  = m_xi;
	__float128 Pi = 3.1415926535897932384626433832795;
	__float128 a = aa;

	// Fill tables
	for ( int kk = 0; kk < nR; kk++ ) 
	{

		// Initialize entries
		h_ewaldC1.data[ 2*kk ].x = 0.0; // UF1
		h_ewaldC1.data[ 2*kk ].y = 0.0; // UF2
		h_ewaldC1.data[ 2*kk ].z = 0.0; // UC1
		h_ewaldC1.data[ 2*kk ].w = 0.0; // UC2 
		h_ewaldC1.data[ 2*kk + 1 ].x = 0.0; // DC1
		h_ewaldC1.data[ 2*kk + 1 ].y = 0.0; // DC2
		h_ewaldC1.data[ 2*kk + 1 ].z = 0.0; // DC3
		h_ewaldC1.data[ 2*kk + 1 ].w = 0.0; // extra 

		// Distance for current entry
		__float128 r = __float128( kk ) * dr + dr;
		__float128 Imrr = 0.00000000000000000000000000000000;
		__float128 rr   = 0.00000000000000000000000000000000;
		__float128 g1   = 0.00000000000000000000000000000000;
		__float128 g2   = 0.00000000000000000000000000000000;
		__float128 h1   = 0.00000000000000000000000000000000;
		__float128 h2   = 0.00000000000000000000000000000000;
		__float128 h3   = 0.00000000000000000000000000000000;
		

		// Expression have been simplified assuming no overlap, touching, and overlap
		if ( r > 2.0*a ){

		  Imrr = -powq(a,-1) + (powq(a,2)*powq(r,-3))/2. + (3*powq(r,-1))/4. + (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*(-12*powq(r,4) + powq(xi,-4)))/128. + 
		    powq(a,-2)*((9*r)/32. - (3*powq(r,-3)*powq(xi,-4))/128.) + 
		    (erfcq((2*a + r)*xi)*(128*powq(a,-1) + 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(36*r - 3*powq(r,-3)*powq(xi,-4))))/256. + 
		    (erfcq(2*a*xi - r*xi)*(128*powq(a,-1) - 64*powq(a,2)*powq(r,-3) - 96*powq(r,-1) + powq(a,-2)*(-36*r + 3*powq(r,-3)*powq(xi,-4))))/
		    256. + (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(1 + 6*powq(r,2)*powq(xi,2)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) - 16*powq(a,3)*powq(xi,2) + a*(2 - 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) + 16*powq(a,3)*powq(xi,2) + a*(-2 + 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128.;

		  rr = -powq(a,-1) - powq(a,2)*powq(r,-3) + (3*powq(r,-1))/2. + (3*powq(a,-2)*powq(r,-3)*(4*powq(r,4) + powq(xi,-4)))/64. + 
		    (erfcq(2*a*xi - r*xi)*(64*powq(a,-1) + 64*powq(a,2)*powq(r,-3) - 96*powq(r,-1) + powq(a,-2)*(-12*r - 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (erfcq((2*a + r)*xi)*(64*powq(a,-1) - 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(12*r + 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(-1 + 2*powq(r,2)*powq(xi,2)))/32. - 
		    ((2*a + 3*r)*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 - 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. + 
		    ((2*a - 3*r)*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 + 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. - 
		    (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*powq(xi,-4)*(1 + 4*powq(r,4)*powq(xi,4)))/64.;

		  g1 = (expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(9 + 15*powq(r,2)*powq(xi,2) - 30*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a - 45*r - 3*(2*a + r)*(-16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) + 6*(2*a + r)*(-32*r*powq(a,3) + 
			32*powq(a,4) + 44*powq(a,2)*powq(r,2) - 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/
		    640. + (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
			    (-9*(2*a + 5*r) + 3*(2*a - r)*(16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) - 6*(2*a - r)*(32*r*powq(a,3) + 
			32*powq(a,4) + 44*powq(a,2)*powq(r,2) + 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/640. + 
		    (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(3 + 3*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 
			512*powq(a,6)*powq(xi,6) - 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/1280. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 
			512*powq(a,6)*powq(xi,6) + 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/1280.;

		  g2 = (-3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(18*a + 45*r - 3*(24*r*powq(a,2) + 16*powq(a,3) + 
		14*a*powq(r,2) + 5*powq(r,3))*powq(xi,2) + 6*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(-2*a + r,2)*powq(xi,4)))/640. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(-18*a + 45*r + 3*(-24*r*powq(a,2) + 16*powq(a,3) + 
		14*a*powq(r,2) - 5*powq(r,3))*powq(xi,2) - 6*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(2*a + r,2)*powq(xi,4)))/640. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 - 15*powq(r,2)*powq(xi,2) + 4*(128*powq(a,6) - 80*powq(a,4)*powq(r,2) + 
		16*a*powq(r,5) - 5*powq(r,6))*powq(xi,6)))/1280. + (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-3 + 3*powq(r,2)*powq(xi,2) + 
		4*powq(r,6)*powq(xi,6)))/128. - (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-15 + 15*powq(r,2)*powq(xi,2) + 4*(-128*powq(a,6) + 
		80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) + 5*powq(r,6))*powq(xi,6)))/1280.;

		  h1 = (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(27 - 2*powq(xi,2)*(15*powq(r,2) + 2*powq(r,4)*powq(xi,2) - 
			4*powq(r,6)*powq(xi,4) + 48*powq(a,2)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))))/4096. + 
			  (3*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(270*a - 135*r + 6*(2*a + 5*r)*(12*powq(a,2) 
			+ 5*powq(r,2))*powq(xi,2) - 4*(144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) - 5*powq(r,5))*powq(xi,4) + 
		      8*powq(2*a - r,3)*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(xi,6)))/40960. + 
		    (3*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(-135*(2*a + r) - 6*(2*a - 5*r)*(12*powq(a,2) + 
			5*powq(r,2))*powq(xi,2) + 4*(-144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) + 5*powq(r,5))*powq(xi,4) - 
		      8*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,3)*powq(xi,6)))/40960. + 
		    (3*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(27 + 8*powq(xi,2)*(-6*powq(r,2) + 9*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
			12*powq(a,2)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192. + (3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*
			powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 16*(96*r*powq(a,3) + 
			48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(-2*a + r,4)*powq(xi,8)))/81920. + 
			(3*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + 
			powq(r,2))*powq(xi,4) + 16*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 
			5*powq(r,4))*powq(2*a + r,4)*powq(xi,8)))/81920.;

		  h2 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(-45 - 78*powq(r,2)*powq(xi,2) + 
			28*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + 19*powq(r,2)*powq(xi,2) + 10*powq(r,4)*powq(xi,4)) - 
			56*powq(r,6)*powq(xi,6)))/4096. + (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(2*a + r) + 
			6*(-20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) + 13*powq(r,3))*powq(xi,2) - 4*(2*a + r)*(-32*r*powq(a,3) + 16*powq(a,4) + 
			48*powq(a,2)*powq(r,2) - 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) + 8*(2*a + r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 
			7*powq(r,4))*powq(-2*a + r,2)*powq(xi,6)))/8192. + (9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*
		     (45*(-2*a + r) - 6*(20*r*powq(a,2) + 8*powq(a,3) + 46*a*powq(r,2) - 13*powq(r,3))*powq(xi,2) + 4*(2*a - r)*(32*r*powq(a,3) + 16*powq(a,4) + 
			48*powq(a,2)*powq(r,2) + 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) - 8*(2*a - r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 
			7*powq(r,4))*powq(2*a + r,2)*powq(xi,6)))/8192. - (9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 
			7*powq(r,8))*powq(xi,6))))/16384. - (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 
		6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 40*powq(a,2)*powq(r,6) + 
			7*powq(r,8))*powq(xi,6))))/16384. - (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(45 + 8*powq(xi,2)*(6*powq(r,2) - 
		9*powq(r,4)*powq(xi,2) - 14*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(-15 - 9*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))))/8192.;

		  h3 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(-45 + 18*powq(r,2)*powq(xi,2) - 
			4*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + powq(r,2)*powq(xi,2) - 2*powq(r,4)*powq(xi,4)) + 
			8*powq(r,6)*powq(xi,6)))/4096. + (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(2*a + r) + 
			6*(2*a - 3*r)*powq(-2*a + r,2)*powq(xi,2) - 4*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
			8*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(2*a + r,2)*powq(xi,6)))/8192. + 
			(9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(-2*a + r) - 
			6*(2*a + 3*r)*powq(2*a + r,2)*powq(xi,2) + 4*(4*powq(a,2) + powq(r,2))*powq(2*a + r,3)*powq(xi,4) - 8*(4*powq(a,2) + 
			powq(r,2))*powq(-2*a + r,2)*powq(2*a + r,3)*powq(xi,6)))/8192. - 
			(9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 
			3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. - 
		    (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 3*powq(r,2)*(12*powq(a,2) + 
			powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. + 
		    (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(6*powq(r,2) - 3*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
			4*powq(a,2)*(15 - 9*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192.;
		}
		else if ( r == 2.0*a ){
				
		  Imrr = -(powq(a,-5)*(3 + 16*a*xi*powq(Pi,-0.5))*powq(xi,-4))/2048. + (3*erfcq(2*a*xi)*powq(a,-5)*(-192*powq(a,4) + powq(xi,-4)))/1024. + 
		    erfcq(4*a*xi)*(powq(a,-1) - (3*powq(a,-5)*powq(xi,-4))/2048.) + 
		    (expq(-16*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(-1 - 64*powq(a,2)*powq(xi,2)))/256. + 
		    (3*expq(-4*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(1 + 24*powq(a,2)*powq(xi,2)))/256.;

		  rr = (powq(a,-5)*(3 + 16*a*xi*powq(Pi,-0.5))*powq(xi,-4))/1024. + erfcq(2*a*xi)*((-3*powq(a,-1))/8. - (3*powq(a,-5)*powq(xi,-4))/512.) + 
		    erfcq(4*a*xi)*(powq(a,-1) + (3*powq(a,-5)*powq(xi,-4))/1024.) + 
		    (expq(-16*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(1 - 32*powq(a,2)*powq(xi,2)))/128. + 
		    (3*expq(-4*powq(a,2)*powq(xi,2))*powq(a,-4)*powq(Pi,-0.5)*powq(xi,-3)*(-1 + 8*powq(a,2)*powq(xi,2)))/128.;

		  g1 = (expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(9 + 15*powq(r,2)*powq(xi,2) - 30*powq(r,4)*powq(xi,4)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(18*a - 45*r - 3*(2*a + r)*(-16*a*r + 8*powq(a,2) + 
			25*powq(r,2))*powq(xi,2) + 6*(2*a + r)*(-32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) - 36*a*powq(r,3) + 
			25*powq(r,4))*powq(xi,4)))/640. + (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(-9*(2*a + 5*r) + 
			3*(2*a - r)*(16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) - 6*(2*a - r)*(32*r*powq(a,3) + 32*powq(a,4) + 44*powq(a,2)*powq(r,2) + 
			36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/640. + (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(3 + 3*powq(r,2)*powq(xi,2) + 
			20*powq(r,6)*powq(xi,6)))/128. - 
			(3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 
			512*powq(a,6)*powq(xi,6) - 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/1280. - 
			(3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 
			512*powq(a,6)*powq(xi,6) + 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/1280.;

		  g2 = (-3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))/64. + 
			  (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(18*a + 45*r - 3*(24*r*powq(a,2) + 
			16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(xi,2) + 6*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 
			5*powq(r,3))*powq(-2*a + r,2)*powq(xi,4)))/640. + 
			  (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(-18*a + 45*r + 3*(-24*r*powq(a,2) + 
				16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(xi,2) - 6*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 
				5*powq(r,3))*powq(2*a + r,2)*powq(xi,4)))/640. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 - 15*powq(r,2)*powq(xi,2) + 4*(128*powq(a,6) - 80*powq(a,4)*powq(r,2) + 
			16*a*powq(r,5) - 5*powq(r,6))*powq(xi,6)))/1280. + 
		    (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-15 + 15*powq(r,2)*powq(xi,2) + 4*(-128*powq(a,6) + 
			80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) + 5*powq(r,6))*powq(xi,6)))/1280.;
		
		  h1 = (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(27 - 2*powq(xi,2)*(15*powq(r,2) + 
			2*powq(r,4)*powq(xi,2) - 4*powq(r,6)*powq(xi,4) + 48*powq(a,2)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))))/4096. + 
		    	(3*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(270*a - 135*r + 
				6*(2*a + 5*r)*(12*powq(a,2) + 5*powq(r,2))*powq(xi,2) - 4*(144*r*powq(a,4) + 96*powq(a,5) + 
				64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) - 5*powq(r,5))*powq(xi,4) + 8*powq(2*a - r,3)*(96*r*powq(a,3) + 48*powq(a,4) + 
				80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(xi,6)))/40960. + 
		    	(3*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(-135*(2*a + r) - 6*(2*a - 5*r)*(12*powq(a,2) + 
				5*powq(r,2))*powq(xi,2) + 4*(-144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) + 
				5*powq(r,5))*powq(xi,4) - 8*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 
				5*powq(r,4))*powq(2*a + r,3)*powq(xi,6)))/40960. + 
		    	(3*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(27 + 8*powq(xi,2)*(-6*powq(r,2) + 9*powq(r,4)*powq(xi,2) - 
				2*powq(r,8)*powq(xi,6) + 12*powq(a,2)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192. + 
		    	(3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 
				360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 16*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 
				40*a*powq(r,3) + 5*powq(r,4))*powq(-2*a + r,4)*powq(xi,8)))/81920. + 
		    	(3*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 
				360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 16*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 
					40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,4)*powq(xi,8)))/81920.;

		  h2 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(-45 - 78*powq(r,2)*powq(xi,2) + 
				28*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + 19*powq(r,2)*powq(xi,2) + 
					10*powq(r,4)*powq(xi,4)) - 56*powq(r,6)*powq(xi,6)))/4096. + 
		    	(9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(2*a + r) + 6*(-20*r*powq(a,2) + 
				8*powq(a,3) + 46*a*powq(r,2) + 13*powq(r,3))*powq(xi,2) - 4*(2*a + r)*(-32*r*powq(a,3) + 16*powq(a,4) + 
				48*powq(a,2)*powq(r,2) - 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) + 8*(2*a + r)*(16*powq(a,4) + 
				16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(-2*a + r,2)*powq(xi,6)))/8192. + 
		    	(9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(-2*a + r) - 6*(20*r*powq(a,2) + 
				8*powq(a,3) + 46*a*powq(r,2) - 13*powq(r,3))*powq(xi,2) + 4*(2*a - r)*(32*r*powq(a,3) + 16*powq(a,4) + 
				48*powq(a,2)*powq(r,2) + 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) - 8*(2*a - r)*(16*powq(a,4) + 
				16*powq(a,2)*powq(r,2) + 7*powq(r,4))*powq(2*a + r,2)*powq(xi,6)))/8192. - 
		    	(9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 
				9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 
					40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. - 
		    	(9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 6*powq(r,2) + 
				9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 
					40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. - 
		    	(9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(45 + 8*powq(xi,2)*(6*powq(r,2) - 9*powq(r,4)*powq(xi,2) - 
				14*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(-15 - 9*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))))/8192.;

		  h3 = (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(-45 + 18*powq(r,2)*powq(xi,2) - 
			4*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + powq(r,2)*powq(xi,2) - 2*powq(r,4)*powq(xi,4)) + 8*powq(r,6)*powq(xi,6)))/4096. + 
		    	(9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(2*a + r) + 
				6*(2*a - 3*r)*powq(-2*a + r,2)*powq(xi,2) - 4*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 
				8*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(2*a + r,2)*powq(xi,6)))/8192. + 
			(9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(-2*a + r) - 
				6*(2*a + 3*r)*powq(2*a + r,2)*powq(xi,2) + 4*(4*powq(a,2) + powq(r,2))*powq(2*a + r,3)*powq(xi,4) - 
				8*(4*powq(a,2) + powq(r,2))*powq(-2*a + r,2)*powq(2*a + r,3)*powq(xi,6)))/8192. - 
			(9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 
				3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + 
					powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. - 
		    	(9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 
				3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + 
					powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. + 
		    	(9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(6*powq(r,2) - 3*powq(r,4)*powq(xi,2) - 
				2*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(15 - 9*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192.;
		}
		else if ( r < 2*a){

		  Imrr = (-9*r*powq(a,-2))/32. + powq(a,-1) - (powq(a,2)*powq(r,-3))/2. - (3*powq(r,-1))/4. + 
		    (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*(-12*powq(r,4) + powq(xi,-4)))/128. + 
		    (erfcq((-2*a + r)*xi)*(-128*powq(a,-1) + 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(36*r - 3*powq(r,-3)*powq(xi,-4))))/
		    256. + (erfcq((2*a + r)*xi)*(128*powq(a,-1) + 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(36*r - 3*powq(r,-3)*powq(xi,-4))))/
		    256. + (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(1 + 6*powq(r,2)*powq(xi,2)))/64. + 
		    (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) - 16*powq(a,3)*powq(xi,2) + a*(2 - 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128. + 
		    (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (8*r*powq(a,2)*powq(xi,2) + 16*powq(a,3)*powq(xi,2) + a*(-2 + 28*powq(r,2)*powq(xi,2)) - 3*(r + 6*powq(r,3)*powq(xi,2))))/128.;

		  rr = ((2*a + 3*r)*powq(a,-2)*powq(2*a - r,3)*powq(r,-3))/16. + 
		    (erfcq((-2*a + r)*xi)*(-64*powq(a,-1) - 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(12*r + 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (erfcq((2*a + r)*xi)*(64*powq(a,-1) - 64*powq(a,2)*powq(r,-3) + 96*powq(r,-1) + powq(a,-2)*(12*r + 3*powq(r,-3)*powq(xi,-4))))/128. + 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-2)*powq(xi,-3)*(-1 + 2*powq(r,2)*powq(xi,2)))/32. - 
		    ((2*a + 3*r)*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 - 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. + 
		    ((2*a - 3*r)*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-2)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-3)*
		     (-1 + 8*a*r*powq(xi,2) + 8*powq(a,2)*powq(xi,2) + 2*powq(r,2)*powq(xi,2)))/64. - 
		    (3*erfcq(r*xi)*powq(a,-2)*powq(r,-3)*powq(xi,-4)*(1 + 4*powq(r,4)*powq(xi,4)))/64.;

		  g1 = (-9*powq(a,-4)*powq(r,-4)*powq(xi,-6))/128. - (9*powq(a,-4)*powq(r,-2)*powq(xi,-4))/128. + 
			  (expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(9 + 15*powq(r,2)*powq(xi,2) - 
			30*powq(r,4)*powq(xi,4)))/64. + (expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
		     (18*a - 45*r - 3*(2*a + r)*(-16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) + 6*(2*a + r)*(-32*r*powq(a,3) + 32*powq(a,4) + 
			44*powq(a,2)*powq(r,2) - 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/640. + 
			(expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(-9*(2*a + 5*r) + 
			3*(2*a - r)*(16*a*r + 8*powq(a,2) + 25*powq(r,2))*powq(xi,2) - 6*(2*a - r)*(32*r*powq(a,3) + 32*powq(a,4) + 
			44*powq(a,2)*powq(r,2) + 36*a*powq(r,3) + 25*powq(r,4))*powq(xi,4)))/640. + 
			(3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(3 + 3*powq(r,2)*powq(xi,2) + 20*powq(r,6)*powq(xi,6)))/128. + 
		    (3*erfcq(2*a*xi - r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 
			512*powq(a,6)*powq(xi,6) - 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/1280. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 + 5*powq(r,2)*powq(xi,2)*(3 + 64*powq(a,4)*powq(xi,4)) + 
			512*powq(a,6)*powq(xi,6) + 256*a*powq(r,5)*powq(xi,6) + 100*powq(r,6)*powq(xi,6)))/1280.;

		  g2 = (-3*r*powq(a,-3))/10. - (12*powq(a,2)*powq(r,-4))/5. + (3*powq(r,-2))/2. + (3*powq(a,-4)*powq(r,2))/32. - 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-3)*powq(xi,-5)*(3 - powq(r,2)*powq(xi,2) + 
			2*powq(r,4)*powq(xi,4)))/64. + (expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*
			(18*a + 45*r - 3*(24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(xi,2) + 6*(24*r*powq(a,2) + 
			16*powq(a,3) + 14*a*powq(r,2) + 5*powq(r,3))*powq(-2*a + r,2)*powq(xi,4)))/640. + 
			(expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-4)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-5)*(-18*a + 45*r + 3*(-24*r*powq(a,2) + 
			16*powq(a,3) + 14*a*powq(r,2) - 5*powq(r,3))*powq(xi,2) - 6*(-24*r*powq(a,2) + 16*powq(a,3) + 14*a*powq(r,2) - 
			5*powq(r,3))*powq(2*a + r,2)*powq(xi,4)))/640. + (3*erfcq((-2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(15 - 15*powq(r,2)*powq(xi,2) + 
			4*(128*powq(a,6) - 80*powq(a,4)*powq(r,2) + 16*a*powq(r,5) - 5*powq(r,6))*powq(xi,6)))/1280. + 
		    (3*erfcq(r*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))/128. - 
		    (3*erfcq((2*a + r)*xi)*powq(a,-4)*powq(r,-4)*powq(xi,-6)*(-15 + 15*powq(r,2)*powq(xi,2) + 4*(-128*powq(a,6) + 80*powq(a,4)*powq(r,2) + 
			16*a*powq(r,5) + 5*powq(r,6))*powq(xi,6)))/1280.;

		  h1 = (9*r*powq(a,-4))/64. - (3*powq(a,-3))/10. - (9*powq(a,2)*powq(r,-5))/10. + (3*powq(r,-3))/4. - (3*powq(a,-6)*powq(r,3))/512. + 
		    (3*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(27 - 2*powq(xi,2)*(15*powq(r,2) + 
		2*powq(r,4)*powq(xi,2) - 4*powq(r,6)*powq(xi,4) + 48*powq(a,2)*(3 - powq(r,2)*powq(xi,2) + 2*powq(r,4)*powq(xi,4)))))/4096. + 
		    (3*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(270*a - 135*r + 6*(2*a + 5*r)*(12*powq(a,2) + 
		5*powq(r,2))*powq(xi,2) - 4*(144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) - 5*powq(r,5))*powq(xi,4) + 
		      8*powq(2*a - r,3)*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 5*powq(r,4))*powq(xi,6)))/40960. + 
		    (3*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(-135*(2*a + r) - 6*(2*a - 5*r)*(12*powq(a,2) + 
		5*powq(r,2))*powq(xi,2) + 4*(-144*r*powq(a,4) + 96*powq(a,5) + 64*powq(a,3)*powq(r,2) - 30*a*powq(r,4) + 5*powq(r,5))*powq(xi,4) - 
		8*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,3)*powq(xi,6)))/40960. + 
		    (3*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(27 + 8*powq(xi,2)*(-6*powq(r,2) + 9*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
		12*powq(a,2)*(-3 + 3*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192. + 
		    (3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + 
		powq(r,2))*powq(xi,4) + 16*(96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) + 40*a*powq(r,3) + 
		5*powq(r,4))*powq(-2*a + r,4)*powq(xi,8)))/81920. + (3*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-135 + 240*(6*powq(a,2) + 
		powq(r,2))*powq(xi,2) - 360*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 16*(-96*r*powq(a,3) + 48*powq(a,4) + 80*powq(a,2)*powq(r,2) - 
		40*a*powq(r,3) + 5*powq(r,4))*powq(2*a + r,4)*powq(xi,8)))/81920.;

		  h2 = (63*r*powq(a,-4))/64. - (3*powq(a,-3))/2. + (9*powq(a,2)*powq(r,-5))/2. - (3*powq(r,-3))/4. - (33*powq(a,-6)*powq(r,3))/512. + 
			  (9*powq(a,-6)*powq(r,-3)*powq(xi,-6))/128. - (27*powq(a,-4)*powq(r,-3)*powq(xi,-4))/64. + 
			  (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(-45 - 78*powq(r,2)*powq(xi,2) + 
			28*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + 19*powq(r,2)*powq(xi,2) + 10*powq(r,4)*powq(xi,4)) - 
			56*powq(r,6)*powq(xi,6)))/4096. + 
		(3*erfcq(2*a*xi - r*xi)*powq(a,-6)*powq(r,-3)*powq(xi,-6)*(-3 + 18*powq(a,2)*powq(xi,2)*(1 - 4*powq(r,4)*powq(xi,4)) 
		+ 128*powq(a,6)*powq(xi,6) + 64*powq(a,3)*powq(r,3)*powq(xi,6) + 8*powq(r,6)*powq(xi,6)))/256. + 
		(9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(2*a + r) + 6*(-20*r*powq(a,2) + 
		8*powq(a,3) + 46*a*powq(r,2) + 13*powq(r,3))*powq(xi,2) - 4*(2*a + r)*(-32*r*powq(a,3) + 16*powq(a,4) + 
		48*powq(a,2)*powq(r,2) - 56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) + 8*(2*a + r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 
		7*powq(r,4))*powq(-2*a + r,2)*powq(xi,6)))/8192. + (
		9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(-2*a + r) - 6*(20*r*powq(a,2) + 
		8*powq(a,3) + 46*a*powq(r,2) - 13*powq(r,3))*powq(xi,2) + 4*(2*a - r)*(32*r*powq(a,3) + 16*powq(a,4) + 48*powq(a,2)*powq(r,2) + 
		56*a*powq(r,3) + 7*powq(r,4))*powq(xi,4) - 8*(2*a - r)*(16*powq(a,4) + 16*powq(a,2)*powq(r,2) + 
		7*powq(r,4))*powq(2*a + r,2)*powq(xi,6)))/8192. - (9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) - 
		6*powq(r,2) + 9*powq(r,2)*(4*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(256*powq(a,8) + 128*powq(a,6)*powq(r,2) - 
		40*powq(a,2)*powq(r,6) + 7*powq(r,8))*powq(xi,6))))/16384. + 
		(3*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(135 + 8*powq(xi,2)*(-6*(30*powq(a,2) + powq(r,2)) + 
		9*(4*powq(a,2) - 3*powq(r,2))*powq(r,2)*powq(xi,2) + 2*(-768*powq(a,8) + 128*powq(a,6)*powq(r,2) + 256*powq(a,3)*powq(r,5) - 
		168*powq(a,2)*powq(r,6) + 11*powq(r,8))*powq(xi,6))))/16384. - (9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(45 + 
		8*powq(xi,2)*(6*powq(r,2) - 9*powq(r,4)*powq(xi,2) - 14*powq(r,8)*powq(xi,6) + 4*powq(a,2)*(-15 - 9*powq(r,2)*powq(xi,2) + 
		20*powq(r,6)*powq(xi,6)))))/8192.;

		  h3 = (9*r*powq(a,-4))/64. + (9*powq(a,2)*powq(r,-5))/2. - (9*powq(r,-3))/4. - (9*powq(a,-6)*powq(r,3))/512. + 
		    (9*expq(-(powq(r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-4)*powq(xi,-7)*(-45 + 18*powq(r,2)*powq(xi,2) - 
		4*powq(r,4)*powq(xi,4) + 32*powq(a,2)*powq(xi,2)*(15 + powq(r,2)*powq(xi,2) - 2*powq(r,4)*powq(xi,4)) + 
		8*powq(r,6)*powq(xi,6)))/4096. + (9*expq(-(powq(2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(2*a + r) + 
		6*(2*a - 3*r)*powq(-2*a + r,2)*powq(xi,2) - 4*powq(2*a - r,3)*(4*powq(a,2) + powq(r,2))*powq(xi,4) + 8*powq(2*a - r,3)*(4*powq(a,2) + 
		powq(r,2))*powq(2*a + r,2)*powq(xi,6)))/8192. + 
		(9*expq(-(powq(-2*a + r,2)*powq(xi,2)))*powq(a,-6)*powq(Pi,-0.5)*powq(r,-5)*powq(xi,-7)*(45*(-2*a + r) - 
		6*(2*a + 3*r)*powq(2*a + r,2)*powq(xi,2) + 4*(4*powq(a,2) + powq(r,2))*powq(2*a + r,3)*powq(xi,4) - 8*(4*powq(a,2) + 
		powq(r,2))*powq(-2*a + r,2)*powq(2*a + r,3)*powq(xi,6)))/8192. - 
		(9*erfcq((-2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 
		3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. - 
		(9*erfcq((2*a + r)*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(60*powq(a,2) + 6*powq(r,2) - 
		3*powq(r,2)*(12*powq(a,2) + powq(r,2))*powq(xi,2) + 2*(4*powq(a,2) + powq(r,2))*powq(4*powq(a,2) - powq(r,2),3)*powq(xi,6))))/16384. + 
		(9*erfcq(r*xi)*powq(a,-6)*powq(r,-5)*powq(xi,-8)*(-45 + 8*powq(xi,2)*(6*powq(r,2) - 3*powq(r,4)*powq(xi,2) - 2*powq(r,8)*powq(xi,6) + 
		4*powq(a,2)*(15 - 9*powq(r,2)*powq(xi,2) + 4*powq(r,6)*powq(xi,6)))))/8192.;

		}

		// Save values to table
		h_ewaldC1.data[ 2*kk ].x = Scalar( Imrr ); // UF1
		h_ewaldC1.data[ 2*kk ].y = Scalar( rr );   // UF2
		h_ewaldC1.data[ 2*kk ].z = Scalar( g1/2. );  // UC1
		h_ewaldC1.data[ 2*kk ].w = Scalar( -g2/2. ); // UC2
		h_ewaldC1.data[ 2*kk + 1 ].x = Scalar( h1 ); // DC1
		h_ewaldC1.data[ 2*kk + 1 ].y = Scalar( h2 ); // DC2
		h_ewaldC1.data[ 2*kk + 1 ].z = Scalar( h3 ); // DC3


	} // kk loop over distances

	// Applied forces/torques 
	// Particle linear/angular velocities, plus stresslet
	unsigned int group_size = m_group_p->getNumMembers();
	unsigned int group_size_b = m_group->getNumMembers();
	unsigned int N = m_pdata->getN();

	GPUArray<Scalar> n_AppliedForce(6*group_size, m_exec_conf);
	GPUArray<Scalar> n_Velocity(   (12*group_size_b), m_exec_conf);
	GPUArray<Scalar> n_Stress(   (22*group_size), m_exec_conf);
	m_AppliedForce.swap(n_AppliedForce);
	m_Velocity.swap(n_Velocity);
	m_Stress.swap(n_Stress);

	// Deepak: added for rigid assemblies
	GPUArray<Scalar3> n_rel_pos(N, m_exec_conf);
	GPUArray<Scalar3> n_rel_pos_bf(N, m_exec_conf);
	GPUArray<int> n_body_tag(N, m_exec_conf);
	GPUArray<int> n_local_index(N, m_exec_conf);
	m_rel_pos.swap(n_rel_pos);
	m_rel_pos_bf.swap(n_rel_pos_bf);
	m_body_tag.swap(n_body_tag);
	m_local_index.swap(n_local_index);
}

/*
	Allocate workspace variables
*/
void Stokes::AllocateWorkSpaces(){
	
	// Set up the arrays and memory for calculation work spaces
	// 
	// Total Memory required for the arrays declared in this function:
	// 	
	//	sizeof(float) = sizeof(int) = 4 bytes
	//	
	//	nnz = 468 * N 
	//	mmax = 100	
	//
	//	Variable		Length		Type
	//	--------		------		----
	//	dot_sum			512		float
	//	bro_ff_psi		3*N		float4
	//	bro_ff_UBreal		3*N		float4
	//	bro_ff_Tm		mmax		float
	//	bro_ff_v		3*N		float4
	//	bro_ff_vj		3*N		float4
	//	bro_ff_vjm1		3*N		float4
	//	bro_ff_Mvj 		3*N		float4
	//	bro_ff_V		3*mmax*N	float4
	//	bro_ff_UB_old		3*N		float4
	//	bro_ff_Mpsi	 	3*N		float4
	//	bro_nf_Tm		m_max 		float
	//	bro_nf_v		6*N		float
	//	bro_nf_V		(mmax+1)*6*N	float
	//	bro_nf_FB_old 		6*N	 	float
	//	bro_nf_psi 		6*N		float
	//	saddle_psi		6*N		float
	//	saddle_posPrime		N		float4
	//	saddle_rhs 		17*N		float
	//	saddle_solution 	17*N		float
	//	mob_couplet		2*N		float4
	//	mob_delu		2*N		float4
	//	mob_vel1		N		float4
	//	mob_vel2		N		float4
	//	mob_delu1		2*N		float4
	//	mob_delu2		2*N		float4
	//	mob_vel			N		float4
	//	mob_AngvelStrain	2*N		float4
	//	mob_net_force		N		float4
	//	mob_TorqueStress	2*N		float4
	//	precond_scratch 	N		int
	//	precond_map 		nnz		int
	//	precond_backup	 	nnz		float
	//
	//				2963*N+712 \approx 2963*N
	//
	//	Total size in bytes: 	11852 * N
	//	Total size in KB:	11.852 * N
	//	Total size in MB:	0.011852 * N
	//
	// Some examples for various numbers of particles
	//
	//	N	Size (MB)	Size (GB)
	//	---	---------	---------
	//	1E2	1.1852		0.0011852
	//	1E3	11.852		0.011852
	//	1E4	118.52		0.11852
	//	1E5	1185.2		1.1852
	//	1E6	11852		11.852
	
	// Get the number of particles
	unsigned int group_size_b = m_group->getNumMembers();
	unsigned int group_size = m_group_p->getNumMembers();
	unsigned int N = m_pdata->getN();

	// Maximum number of iterations in the Lanczos method
	int m_max = 100;

	//zhoge: 11N for the real space noise, 6NxNyNz for the wave sapce noise
	cudaMalloc( (void**)&m_work_bro_gauss,		(11*group_size) * sizeof(Scalar) );  

	// Variables for far-field Lanczos iteration
	cudaMalloc( (void**)&m_work_bro_ff_psi,	3*group_size  * sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_bro_ff_UBreal,	3*group_size * sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_bro_ff_Mpsi, 	3*group_size * sizeof(Scalar4) );
	//zhoge: change to simple float/Scalar for the far-field Chow & Saad
	cudaMalloc( (void**)&m_work_bro_ff_v,           11*group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_ff_Tm,		m_max * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_ff_V1,		(m_max+1) * 11 * group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_ff_UB_new1,	            11 * group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_ff_UB_old1,	            11 * group_size * sizeof(Scalar) );

	//zhoge: RFD storage (Brownian drift)
	cudaMalloc( (void**)&m_work_rfd_rhs, (11*group_size+6*group_size_b)*sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_rfd_sol, (11*group_size+6*group_size_b)*sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_saddle_psi,         6*group_size_b*sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_saddle_posPrime,    N*sizeof(Scalar4) );

	// Variables for near-field Lanczos iteration
	cudaMalloc( (void**)&m_work_bro_nf_v,           6*group_size * sizeof(Scalar) );	
	cudaMalloc( (void**)&m_work_bro_nf_Tm,		m_max * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_nf_V,		(m_max+1) * 6*group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_nf_FB_old, 	6*group_size * sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_bro_nf_psi, 	6*group_size*sizeof(Scalar) );

	cudaMalloc( (void**)&m_work_saddle_rhs, 	(11*group_size+6*group_size_b)*sizeof(Scalar) );
	cudaMalloc( (void**)&m_work_saddle_solution, 	(11*group_size+6*group_size_b)*sizeof(Scalar) );

	cudaMalloc( (void**)&m_work_mob_couplet,	2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_delu,		2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_vel1,		group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_vel2,		group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_delu1,		2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_delu2,		2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_vel,		group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_AngvelStrain,	2*group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_net_force,	group_size*sizeof(Scalar4) );
	cudaMalloc( (void**)&m_work_mob_TorqueStress,	2*group_size*sizeof(Scalar4) );

	cudaMalloc( (void**)&m_work_precond_scratch, 	group_size_b*sizeof(int) );	
	cudaMalloc( (void**)&m_work_precond_map, 	m_nnz*sizeof(int) );
	cudaMalloc( (void**)&m_work_precond_backup, 	m_nnz*sizeof(Scalar) );

}


/*
	Free workspace variables
*/
void Stokes::FreeWorkSpaces(){
	
	// Dot product partial sum
	cudaFree( m_work_bro_gauss );  //zhoge

	// Variables for far-field Lanczos iteration	
	cudaFree( m_work_bro_ff_psi );
	cudaFree( m_work_bro_ff_UBreal );
	cudaFree( m_work_bro_ff_Mpsi );
	//zhoge
	cudaFree( m_work_bro_ff_v );
	cudaFree( m_work_bro_ff_Tm );
	cudaFree( m_work_bro_ff_V1 );
	cudaFree( m_work_bro_ff_UB_new1 );
	cudaFree( m_work_bro_ff_UB_old1 );

	cudaFree( m_work_rfd_rhs );
	cudaFree( m_work_rfd_sol );


	// Variables for near-field Lanczos iteration	
	cudaFree( m_work_bro_nf_v );
	cudaFree( m_work_bro_nf_Tm );
	cudaFree( m_work_bro_nf_V );
	cudaFree( m_work_bro_nf_FB_old );
	cudaFree( m_work_bro_nf_psi );

	cudaFree( m_work_saddle_psi );
	cudaFree( m_work_saddle_posPrime );
	cudaFree( m_work_saddle_rhs );
	cudaFree( m_work_saddle_solution );

	cudaFree( m_work_mob_couplet );
	cudaFree( m_work_mob_delu );
	cudaFree( m_work_mob_vel1 );
	cudaFree( m_work_mob_vel2 );
	cudaFree( m_work_mob_delu1 );
	cudaFree( m_work_mob_delu2 );
	cudaFree( m_work_mob_vel );
	cudaFree( m_work_mob_AngvelStrain );
	cudaFree( m_work_mob_net_force );
	cudaFree( m_work_mob_TorqueStress );

	cudaFree( m_work_precond_scratch );
	cudaFree( m_work_precond_map );
	cudaFree( m_work_precond_backup );
}

/*
	Modify entries in the resistance table by the specified friction type

	NOTE: This function must be called AFTER setResistanceTable()

	Friction table contains entries in the following order: (This is the order given in Stokes_ResistanceTable.cc)
		1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22
		XA11 XA12 YA11 YA12 YB11 YB12 XC11 XC12 YC11 YC12 XG11 XG12 YG11 YG12 YH11 YH12 XM11 XM12 YM11 YM12 ZM11 ZM12

	friction_type	string specifying type of friction to add
	h0 		Maximum distance for frictional contact
	alpha		list of strengths of frictional contact

*/
void Stokes::setFriction() {

	// Get handles to the resistance data
    	ArrayHandle<Scalar> h_ResTable_dist(m_ResTable_dist, access_location::host, access_mode::read);
    	ArrayHandle<Scalar> h_ResTable_vals(m_ResTable_vals, access_location::host, access_mode::readwrite);

	// Loop over all distances in array
	for (int ii = 0; ii < 1000; ++ii )
	{
		
		// Current gap width
		Scalar h = h_ResTable_dist.data[ii] - 2.0;

		// If current distance is less than frictional distance, add to it
		if ( h <= m_h0 ){

			// Powers of h and h0
			Scalar h2 = h * h;

			Scalar h02 = m_h0 * m_h0;
			Scalar h03 = m_h0 * h02;

			// Friction coefficient
			Scalar  coeff = m_alpha * ( 2.0 / h03 * h2 - 3.0 / h02 * h + 1.0 / h );


			// Add to arrays. Minus signs account for sign of coefficients (add in magnitude)
			int curr_offset = 22 * ii;

			h_ResTable_vals.data[ curr_offset + 2  ] += coeff;
			h_ResTable_vals.data[ curr_offset + 3  ] -= coeff;
			h_ResTable_vals.data[ curr_offset + 12  ] += 0.5 * coeff;
			h_ResTable_vals.data[ curr_offset + 13  ] -= 0.5 * coeff;
			h_ResTable_vals.data[ curr_offset + 18  ] += 9.0/20.0 * coeff;
			h_ResTable_vals.data[ curr_offset + 19  ] += 9.0/20.0 * coeff;
			h_ResTable_vals.data[ curr_offset + 20  ] += 9.0/20.0 * coeff;
			h_ResTable_vals.data[ curr_offset + 21  ] += 9.0/20.0 * coeff;
		}

	}


}

/*
Write quantities to file

Modified from code written by Zach Sherman in his Immersed Boundary code
*/
void Stokes::OutputData( uint64_t timestep ){

        BoxDim global_box = m_pdata->getGlobalBox();
        Scalar3 L = global_box.getL();
        Scalar volume = L.x * L.y * L.z;
        Scalar coeff = 6.0 * 3.14159 / volume;

    	// Access needed data
        unsigned int group_size_b = m_group->getNumMembers();
        unsigned int group_size = m_group_p->getNumMembers();
	ArrayHandle<Scalar> h_Stress(m_Stress, access_location::host, access_mode::read);

        //Initialize stress tensor
        Scalar Stress[6] = {0};

        // Summation of Brownian stress
        for (unsigned int ii = 0; ii < group_size; ii++) {
                for (unsigned int jj = 0; jj < 5; jj++){
                        Stress[jj] += h_Stress.data[ 5*ii + jj]*coeff;
                }
        }

	stressfile <<  timestep << "," << Stress[0] << "," << Stress[1] << "," << Stress[2] << "," << Stress[3] << "," << Stress[4];

        //Summation of hydrodynamic stress
        std::fill(Stress, Stress + 6, 0);;
        for (unsigned int ii = 0; ii < group_size; ii++) {
		for (unsigned int jj = 0; jj < 5; jj++){
			Stress[jj] += h_Stress.data[ 5*group_size + 5*ii + jj]*coeff;
		}
        }
        stressfile << "," << Stress[0] << "," << Stress[1] << "," << Stress[2] << "," << Stress[3] << "," << Stress[4];

        //Summation of constraint stress
        std::fill(Stress, Stress + 6, 0);;
        for (unsigned int ii = 0; ii < group_size; ii++) {
		for (unsigned int jj = 0; jj < 5; jj++){
			Stress[jj] += h_Stress.data[ 10*group_size + 5*ii + jj]*coeff;
		}
        }
        stressfile << "," << Stress[0] << "," << Stress[1] << "," << Stress[2] << "," << Stress[3] << "," << Stress[4];

        //Summation of VDW+contact stress
        std::fill(Stress, Stress + 6, 0);
        Scalar h_min = 0;
        unsigned int overlap_count = 0;
        Scalar h_avg = 0.0;
        for (unsigned int ii = 0; ii < group_size; ii++) {
		for (unsigned int jj = 0; jj < 6; jj++){
			Stress[jj] += h_Stress.data[ 15*group_size + 6*ii + jj]*coeff;
		}
		if(h_Stress.data[ 21*group_size +ii]>0){
		        overlap_count += 1;
		        h_avg += h_Stress.data[21*group_size +ii];
		}
		if(h_Stress.data[21*group_size +ii]>h_min) h_min = h_Stress.data[21*group_size +ii];
        }
        if(overlap_count>0) h_avg /= overlap_count;
        stressfile << "," << Stress[0] << "," << Stress[1] << "," << Stress[2] << "," << Stress[3] << "," << Stress[4] << "," 
		<< Stress[5] << "," << h_min << "," << h_avg << "," << overlap_count << std::endl;

}


/* 
  Run the integration method.
  Particle positions and velocities are moved forward to timestep+1 
*/
void Stokes::integrateStepOne(uint64_t timestep)
{

  Scalar kT = m_T->operator()(timestep);
  unsigned int group_size = m_group_p->getNumMembers();
  unsigned int group_size_b = m_group->getNumMembers();

  // Consistency check
  assert(group_size <= m_pdata->getN());
  if (group_size == 0) return;
	
  BoxDim box = m_pdata->getBox();
  // Calculate the shear rate of the current timestep
  Scalar current_shear_rate = m_shear_func -> getShearRate(static_cast<unsigned int>(timestep));

  // Recompute neighbor lists ( if needed )	
  m_nlist_ewald->compute(timestep);

  {
  ArrayHandle<Scalar3> h_rel_pos(m_rel_pos, access_location::host, access_mode::overwrite);
  memset((void*)h_rel_pos.data, 0, sizeof(Scalar3) * m_pdata->getN());

  ArrayHandle<int> h_body_tag(m_body_tag, access_location::host, access_mode::overwrite);
  memset((void*)h_body_tag.data, -1, sizeof(int) * m_pdata->getN());

  ArrayHandle<int> h_local_index(m_local_index, access_location::host, access_mode::overwrite);
  memset((void*)h_local_index.data, -1, sizeof(int) * m_pdata->getN());
  }

  // *****************************
  // Get Handles to Device Arrays
  // *****************************
{
  // Neighbor lists
  ArrayHandle<unsigned int> d_nneigh_ewald(   m_nlist_ewald->getNNeighArray(), access_location::device, access_mode::read );
  ArrayHandle<unsigned int> d_nlist_ewald(    m_nlist_ewald->getNListArray(),  access_location::device, access_mode::read );
  ArrayHandle<size_t> d_headlist_ewald( m_nlist_ewald->getHeadList(),    access_location::device, access_mode::read );
	
  // Pruned neighbor list for lubrication preconditioner (constructed in Precondition_Wrap)
  ArrayHandle<unsigned int> d_nneigh_pruned(   m_nneigh_pruned,   access_location::device, access_mode::readwrite );
  ArrayHandle<unsigned int> d_nlist_pruned(    m_nlist_pruned,    access_location::device, access_mode::readwrite );
  ArrayHandle<unsigned int> d_headlist_pruned( m_headlist_pruned, access_location::device, access_mode::readwrite );

  ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite); 

  // Grid vectors
  ArrayHandle<Scalar4>      d_gridk(m_gridk, access_location::device, access_mode::readwrite); //write if deform
  ArrayHandle<CUFFTCOMPLEX> d_gridX(m_gridX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridY(m_gridY, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridZ(m_gridZ, access_location::device, access_mode::read); 
	
  ArrayHandle<CUFFTCOMPLEX> d_gridXX(m_gridXX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridXY(m_gridXY, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridXZ(m_gridXZ, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridYX(m_gridYX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridYY(m_gridYY, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridYZ(m_gridYZ, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridZX(m_gridZX, access_location::device, access_mode::read); 
  ArrayHandle<CUFFTCOMPLEX> d_gridZY(m_gridZY, access_location::device, access_mode::read); 

  // Real space interaction tabulation
  ArrayHandle<Scalar4> d_ewaldC1(m_ewaldC1, access_location::device, access_mode::read);

  // Lubrication calculation stuff
  ArrayHandle<int>   d_L_RowInd( m_L_RowInd, access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_L_RowPtr( m_L_RowPtr, access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_L_ColInd( m_L_ColInd, access_location::device, access_mode::overwrite ); 
  ArrayHandle<Scalar> d_L_Val(    m_L_Val,    access_location::device, access_mode::overwrite ); 	
  ArrayHandle<Scalar> d_Diag(     m_Diag,     access_location::device, access_mode::overwrite ); 
  ArrayHandle<int>   d_HasNeigh( m_HasNeigh, access_location::device, access_mode::overwrite );

  ArrayHandle<Scalar> d_ResTable_dist( m_ResTable_dist, access_location::device, access_mode::read );
  ArrayHandle<Scalar> d_ResTable_vals( m_ResTable_vals, access_location::device, access_mode::read );
	
  ArrayHandle<unsigned int> d_nneigh_less( m_nneigh_less, access_location::device, access_mode::overwrite ); 
  ArrayHandle<unsigned int> d_NEPP(        m_NEPP,        access_location::device, access_mode::overwrite ); 
  ArrayHandle<unsigned int> d_offset(      m_offset,      access_location::device, access_mode::overwrite ); 

  ArrayHandle<Scalar> d_Scratch1( m_Scratch1, access_location::device, access_mode::overwrite ); 
  ArrayHandle<Scalar> d_Scratch2( m_Scratch2, access_location::device, access_mode::overwrite ); 
  ArrayHandle<Scalar> d_Scratch3( m_Scratch3, access_location::device, access_mode::overwrite );
  ArrayHandle<Scalar> d_Scratch4( m_Scratch4, access_location::device, access_mode::overwrite );
  ArrayHandle<Scalar> d_Scratch5( m_Scratch5, access_location::device, access_mode::overwrite );
  ArrayHandle<Scalar> d_Scratch6( m_Scratch6, access_location::device, access_mode::overwrite );
  ArrayHandle<Scalar> d_Scratch7( m_Scratch7, access_location::device, access_mode::overwrite );
  ArrayHandle<Scalar> d_Scratch8( m_Scratch8, access_location::device, access_mode::overwrite );
  ArrayHandle<Scalar> d_EinfForce( m_EinfForce, access_location::device, access_mode::overwrite );
  ArrayHandle<int>   d_prcm(     m_prcm,     access_location::device, access_mode::overwrite );  

  // Particle index (may change in time, unlike tag)
  ArrayHandle<unsigned int> d_index_array(m_group_p->getIndexArray(), access_location::device, access_mode::read);
  ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
  ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

  // Particle position and orientation
  ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),     access_location::device, access_mode::readwrite );
  ArrayHandle<Scalar4> d_ori(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite ); //zhoge:temp 
  ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite );
  ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite );
  ArrayHandle<Scalar4> d_net_force( m_pdata->getNetForce(), access_location::device, access_mode::read );
  ArrayHandle<Scalar4> d_net_torque( m_pdata->getNetTorqueArray(), access_location::device, access_mode::read );
  
  // Linear/angular velocities and applied force/torque
  ArrayHandle<Scalar> d_Velocity(    m_Velocity,     access_location::device, access_mode::readwrite); 
  ArrayHandle<Scalar> d_Stress(    m_Stress,     access_location::device, access_mode::readwrite);
  ArrayHandle<Scalar> d_AppliedForce(m_AppliedForce, access_location::device, access_mode::readwrite); 

  // Rotational noise
//   ArrayHandle<Scalar3> d_noise_ang(m_noise_ang, access_location::device, access_mode::read);
  
  //Deepak:added for rigid
  ArrayHandle<Scalar3> d_rel_pos(m_rel_pos, access_location::device, access_mode::readwrite);
  ArrayHandle<Scalar3> d_rel_pos_bf(m_rel_pos_bf, access_location::device, access_mode::readwrite);
  ArrayHandle<int> d_body_tag(m_body_tag, access_location::device, access_mode::readwrite);
  ArrayHandle<int>   d_local_index(m_local_index, access_location::device, access_mode::readwrite );
  ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),access_location::device, access_mode::read);

  // ***********************
  // Set up data structures
  // ***********************

  // Initialize values in the data structure for Brownian calculation
  BrownianData bro_struct = {
			     m_error,  //tol
			     timestep,
			     m_seed_ff_rs,
			     m_seed_ff_ws,
			     m_seed_nf,
			     m_seed_rfd,
			     m_m_Lanczos_ff,
			     m_m_Lanczos_nf,
			     kT,
			     m_rfd_epsilon,
			     m_work_rfd_rhs,
			     m_work_rfd_sol
  };
  BrownianData *bro_data = &bro_struct;

  // Initialize values in the data structure for mobility calculations
  MobilityData mob_struct = {
			     m_xi,
			     m_ewald_cut,
			     m_ewald_dr,
			     m_ewald_n,
			     d_ewaldC1.data,
			     m_self,
			     d_nneigh_ewald.data,
			     d_nlist_ewald.data,
			     d_headlist_ewald.data,
			     m_eta,
			     m_gaussP,
			     m_gridh,
			     d_gridk.data,
			     d_gridX.data,
			     d_gridY.data,
			     d_gridZ.data,
			     d_gridXX.data,
			     d_gridXY.data,
			     d_gridXZ.data,
			     d_gridYX.data,
			     d_gridYY.data,
			     d_gridYZ.data,
			     d_gridZX.data,
			     d_gridZY.data,
			     plan,
			     m_Nx,
			     m_Ny,
			     m_Nz
  };
  MobilityData *mob_data = &mob_struct;

  // Initialize values in the data structure for the resistance calculations
  //
  // !!! The pointers to the neighbor list here are the EXACT SAME pointers
  //     used mobility structure. Replicated here for simplicity because
  //     it's only ever read, not modified. This way also leaves the option
  //     open to add different neighbor list structures for the lubrication
  //     and mobility calculations. 
  ResistanceData res_struct = {
					4.0,  // lubrication cutoff, rlub 
					2.1,  // lubrication (preconditioner) cutoff, rp
					d_nneigh_ewald.data,
					d_nlist_ewald.data,
					d_headlist_ewald.data,
					d_nneigh_pruned.data,
					d_headlist_pruned.data,
					d_nlist_pruned.data,
					m_nnz,
					d_nneigh_less.data,
					d_NEPP.data,
					d_offset.data,
					d_L_RowInd.data,
					d_L_RowPtr.data,
					d_L_ColInd.data,
					d_L_Val.data,
					d_ResTable_dist.data,
					d_ResTable_vals.data,
					m_ResTable_min,
					m_ResTable_dr,
					soHandle,
					spHandle,
					spStatus,
					descr_R,
					descr_L,
					trans_L,
					trans_Lt,
					info_R,
					info_L,
					info_Lt,
					policy_R,
					policy_L,
					policy_Lt,
					m_pBufferSize,
					d_Scratch1.data,
					d_Scratch2.data,
					d_Scratch3.data,
					d_Scratch4.data,
					d_Scratch5.data,
					d_Scratch6.data,
					d_Scratch7.data,
					d_Scratch8.data,
					d_EinfForce.data,
					d_prcm.data,
					d_HasNeigh.data,
					d_Diag.data,
					m_ichol_relaxer,
					false,
					m_F_rep,    //Repulsive force magnitude
					m_F_att,		//VDW force magnitude
					m_k_n,     //collision spring const
					m_kappa,   //inverse Debye length for electrostatic repulsion
					m_epsq,     //square of the vdw regularization
					m_rcut,
					d_local_index.data,
					d_rel_pos.data, //Deepak:added for rigid
					d_rel_pos_bf.data,
					d_body_tag.data, //Deepak:added for rigid
					d_tag.data,
					d_rtag.data,
					static_cast<unsigned int>(group_size/group_size_b), //Deepak:added for rigid
					group_size_b,
					0,
					max_contact,
					m_contact_table,
					m_k_t,
					m_muf
					};
  ResistanceData *res_data = &res_struct;

  // Initialize values in workspace data
  WorkData work_struct = {
				blasHandle,      //zhoge: was in res_data
				m_work_bro_gauss,  //zhoge: Gaussian random variables
				m_work_bro_ff_psi,
				m_work_bro_ff_UBreal,
				m_work_bro_ff_Mpsi,
				m_work_bro_ff_v,
				m_work_bro_ff_Tm,
				m_work_bro_ff_V1,
				m_work_bro_ff_UB_new1,
				m_work_bro_ff_UB_old1,
				m_work_bro_nf_v,
				m_work_bro_nf_Tm,
				m_work_bro_nf_V,
				m_work_bro_nf_FB_old,
				m_work_bro_nf_psi,
				m_work_saddle_psi,
				m_work_saddle_posPrime,
				m_work_saddle_rhs,
				m_work_saddle_solution,
				m_work_mob_couplet,
				m_work_mob_delu,
				m_work_mob_vel1,
				m_work_mob_vel2,
				m_work_mob_delu1,
				m_work_mob_delu2,
				m_work_mob_vel,
				m_work_mob_AngvelStrain,
				m_work_mob_net_force,
				m_work_mob_TorqueStress,
				m_work_precond_scratch,
				m_work_precond_map,
				m_work_precond_backup
  				};
  WorkData *work_data = &work_struct;


  // Time-dependent external torque (constant if m_omega_ext == 0)
  Scalar T_ext = m_T_ext * cos(m_omega_ext * Scalar(timestep) * m_deltaT);
  
  // *********************************************
  // Perform the update on the GPU (in Stokes.cu)
  // *********************************************
  m_tuner->begin();
  Stokes_StepOne( timestep,
                  m_period,
                  d_pos.data,            //input/output
                  d_ori.data,            //input/output: orientation
                  d_vel.data,
                  d_angmom.data,
                  d_net_force.data,
                  d_net_torque.data,
                  d_AppliedForce.data,   //input/output
                  d_Velocity.data,       //input/output: FSD velocity and stresslet (11N)
                  d_Stress.data,
                  T_ext,                 //external torque
                  m_F_ext,
                  m_deltaT,              //dt       
                  current_shear_rate,
		  delta,
                  256,                    //cuda block size
                  d_image.data,
                  d_index_array.data,
                  d_body.data,
                  group_size,
                  box,
                  bro_data,
                  mob_data,
                  res_data,
                  work_data,
				  m_fric
                  );
  m_tuner->end();
	
  // Save the number of iterations, but reset every so often so that it doesn't grow too large   //zhoge: chow & saad
  m_m_Lanczos_ff = ( ( timestep % 100 == 0 ) || ( bro_data->m_Lanczos_ff > 50 ) ) ? 3 : bro_data->m_Lanczos_ff; 
  m_m_Lanczos_nf = ( ( timestep % 100 == 0 ) || ( bro_data->m_Lanczos_nf > 50 ) ) ? 3 : bro_data->m_Lanczos_nf; 
  
  // Save the relaxation constant, but reset every so often (used in Precondition.cu for the Cholesky decomposition)
  //m_ichol_relaxer = ( ( timestep % 5 == 0 ) || ( m_ichol_relaxer > 1024.0 ) ) ? 1.0 : res_data->ichol_relaxer;
}

  {
    // Output if the period is set (*after* updating, because the indices may change at the next time step)
    if ( ( m_period > 0 ) && ( int(timestep+1) % m_period == 0 ) ) {
      OutputData(int(timestep+1));
    }
  }
  if (m_exec_conf->isCUDAErrorCheckingEnabled())
    CHECK_CUDA_ERROR();
}

/*! \param timestep Current time step
	\post Nothing is done.
*/
void Stokes::integrateStepTwo(uint64_t timestep)
{
}

void export_Stokes(pybind11::module& m)
{
    pybind11::class_<Stokes,IntegrationMethodTwoStep,std::shared_ptr<Stokes>>(m, "Stokes")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
			    std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<NeighborList>,
                            Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar,
                            Scalar, Scalar, Scalar, Scalar, Scalar, int, unsigned int,
			    Scalar, Scalar, std::string, Scalar, Scalar, Scalar, Scalar>())
        .def_property("kT", &Stokes::getT, &Stokes::setT)
        .def("setParams", &Stokes::setParams)
	.def("setShear", &Stokes::setShear)
	.def("OutputData", &Stokes::OutputData)
        .def("setResistanceTable", &Stokes::setResistanceTable)
        .def("setSparseMath", &Stokes::setSparseMath)
        .def("AllocateWorkSpaces", &Stokes::AllocateWorkSpaces)
        .def("setFriction", &Stokes::setFriction)
	.def("setTrigger", &Stokes::setTrigger);
}

#ifdef WIN32
#pragma warning( pop )
#endif
