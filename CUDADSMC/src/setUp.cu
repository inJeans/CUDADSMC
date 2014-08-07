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
                                    int      numberOfAtoms,
									double   Temp,
									double   dBdz,
									curandStatePhilox4_32_10_t *rngState) {
    
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		
		/* Copy state to local memory for efficiency */
		curandStatePhilox4_32_10_t localrngState = rngState[atom];
		
        pos[atom] = selectAtomInDistribution( dBdz, Temp, &localrngState );
		
		vel[atom] = getRandomVelocity( Temp, &localrngState );
		
		// Copy state back to global memory
		rngState[atom] = localrngState;
    }
    return;
}

__device__ double4 operator* ( double a, double4 b )
{
	return make_double4( a*b.x, a*b.y, a*b.z, a*b.w );
}

__device__ double4 getRandomVelocity( double Temp, curandStatePhilox4_32_10_t *rngState )
{
	double4 vel = make_double4( 0., 0., 0., 0. );
	
	double V = sqrt(2.0/3.0*d_kB*Temp/d_mRb);
	
	vel = V * getRandomPointOnUnitSphere( &rngState[0] );
	
	return vel;
}

__device__ double4 getRandomPointOnUnitSphere( curandStatePhilox4_32_10_t *rngState )
{
	double4 pointOnSphere;
	
	double2 r1 = curand_uniform2_double ( &rngState[0] );
	double2 r2 = curand_uniform2_double ( &rngState[0] );
	
	pointOnSphere.x = (double) sqrt(-2.0*log(r1.x)) * sin(2*d_pi*r1.y);
	pointOnSphere.y = (double) sqrt(-2.0*log(r2.x)) * cos(2*d_pi*r2.y);
	pointOnSphere.z = (double) sqrt(-2.0*log(r2.x)) * sin(2*d_pi*r2.y);
	pointOnSphere.w = (double) 0.0;
	
	return pointOnSphere;
}

__device__ double4 selectAtomInDistribution( double dBdz, double Temp, curandStatePhilox4_32_10_t *rngState )
{
    double4 pos = make_double4( 0., 0., 0., 0. );
    double4 r   = make_double4( 0., 0., 0., 0. );

    double meanx = 0.0;
    double stdx  = sqrt( log( 4. ) )*d_kB*Temp / ( d_gs*d_muB*dBdz );
    
    bool noAtomSelected = true;
    
    while (noAtomSelected) {
        
        r = getGaussianPoint( meanx, stdx, &rngState[0] );
        
        if ( pointIsInDistribution( r, dBdz, Temp, &rngState[0] ) ) {
            
            pos = r;
            
            noAtomSelected = false;
        }
    }
    
    return pos;
}

__device__ double2 operator* ( double2 a, double b )
{
	return make_double2( a.x*b, a.y*b );
}

__device__ double2 operator+ ( double2 a, double b )
{
	return make_double2( a.x+b, a.y+b );
}

__device__ double4 getGaussianPoint( double mean, double std, curandStatePhilox4_32_10_t *rngState )
{
    double2 r1 = curand_normal2_double ( &rngState[0] ) * std + mean;
	double2 r2 = curand_normal2_double ( &rngState[0] ) * std + mean;
 
    double4 point = make_double4( r1.x, r1.y, r2.x, r2.y );
    
    return point;
}

__device__ bool pointIsInDistribution( double4 point, double dBdz, double Temp, curandStatePhilox4_32_10_t *rngState )
{
    bool pointIsIn = false;
    
    double potential   = 0.5*d_gs*d_muB*dBdz*sqrt( point.x*point.x + point.y*point.y + 4.*point.z*point.z );
    double probability = exp( -potential / d_kB / Temp );
    
    if ( curand_uniform_double ( &rngState[0] ) < probability ) {
        pointIsIn = true;
    }
    
    return pointIsIn;
}