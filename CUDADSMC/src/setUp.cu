//
//  setUp.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include "setUp.cuh"
#include "math.h"

////////////////////////////////////////////////////////////////////////////////
// Define some global variables on the device                                 //
////////////////////////////////////////////////////////////////////////////////

__constant__ double d_gs   =  0.5;				// Gyromagnetic ratio
__constant__ double d_MF   = -1.0;				// Magnetic quantum number
__constant__ double d_muB  = 9.27400915e-24;	// Bohr magneton
__constant__ double d_mRb  = 1.443160648e-25;	// 87Rb mass
__constant__ double d_pi   = 3.14159265;		// Pi
__constant__ double d_a    = 5.3e-9;			// Constant cross-section formula
__constant__ double d_kB   = 1.3806503e-23;		// Boltzmann's Constant
__constant__ double d_hbar = 1.05457148e-34;	// hbar

////////////////////////////////////////////////////////////////////////////////

#pragma mark - Random Number Generator
__global__ void initRNG( curandStatePhilox4_32_10_t *rngState, int numberOfAtoms )
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		// Each thread gets the same seed, a different sequence
		// number and no offset
		curand_init( 1234, atom, 0, &rngState[atom] );
	}
	
	return;
}

#pragma mark - Initial Distribution

// Kernel to generate the initial distribution
__global__ void generateInitialDist(double4 *pos,
                                    double4 *vel,
									double4 *accel,
                                    int      numberOfAtoms,
									double   Temp,
									double   dBdz,
									curandStatePhilox4_32_10_t *rngState) {
    
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		
		double r1, r2, r3, U, Pr;        // Variables required to generate a position
		bool  noAtomSelected = true;    // Flag to indicate wether we have generated
		// a good position
		
		double meanr = 6*d_kB*Temp/(dBdz*d_gs*d_muB);
		
		/* Copy state to local memory for efficiency */
		curandStatePhilox4_32_10_t localrngState = rngState[atom];
		
		while (noAtomSelected) {
			
			r1 = curand_uniform_double (&localrngState) * 10*meanr - 0.5*10*meanr;
			r2 = curand_uniform_double (&localrngState) * 10*meanr - 0.5*10*meanr;
			r3 = curand_uniform_double (&localrngState) * 10*meanr - 0.5*10*meanr;
			U  = -0.5*d_gs*d_muB*dBdz*sqrt(r1*r1+r2*r2+4.0*r3*r3);
			Pr = exp(U / d_kB / Temp );
			
			if (curand_uniform_double (&localrngState) < Pr) {
				
				pos[atom].x = r1;
				pos[atom].y = r2;
				pos[atom].z = r3;
				pos[atom].w = 1.0;
				
				noAtomSelected = false;
			}
		}
		
		vel[atom] = getRandomVelocity( Temp, localrngState );
		
		// Copy state back to global memory
		rngState[atom] = localrngState;
    }
    return;
}

__device__ double4 operator* ( double a, double4 b )
{
	return make_double4( a*b.x, a*b.y, a*b.z, a*b.w );
}

__device__ double4 getRandomVelocity( double Temp, curandStatePhilox4_32_10_t rngState )
{
	double4 vel = make_double4( 0., 0., 0., 0. );
	
	double V = sqrt(2.0/3.0*d_kB*Temp/d_mRb);
	
	vel = V * getRandomNumberOnUnitSphere( &rngState );
	
	return vel;
}

__device__ double4 getRandomNumberOnUnitSphere( curandStatePhilox4_32_10_t *rngState )
{
	double4 pointOnSphere;
	
	double2 r1 = curand_uniform2_double (&rngState[0]);
	double2 r2 = curand_uniform2_double (&rngState[0]);
	
	pointOnSphere.x = (double) sqrt(-2.0*log(r1.x)) * sin(2*d_pi*r1.y);
	pointOnSphere.y = (double) sqrt(-2.0*log(r2.x)) * cos(2*d_pi*r2.y);
	pointOnSphere.z = (double) sqrt(-2.0*log(r2.x)) * sin(2*d_pi*r2.y);
	pointOnSphere.w = (double) 0.0;
	
	return pointOnSphere;
}