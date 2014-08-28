//
//  setUp.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include "setUp.cuh"
#include "vectorMath.cuh"
#include "math.h"
#include "cudaHelpers.cuh"

#include "declareInitialSystemParameters.cuh"
#include "declareDeviceSystemParameters.cuh"

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
__global__ void generateInitialDist(double3 *pos,
                                    double3 *vel,
                                    double3 *acc,
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
        
        acc[atom] = updateAccel( pos[atom] );
		
		// Copy state back to global memory
		rngState[atom] = localrngState;
    }
    return;
}

__device__ double3 getRandomVelocity( double Temp, curandStatePhilox4_32_10_t *rngState )
{
	double3 vel = make_double3( 0., 0., 0. );
	
	double V = sqrt(2.0/3.0*d_kB*Temp/d_mRb);
	
	vel = V * getRandomPointOnUnitSphere( &rngState[0] );
	
	return vel;
}

__device__ double3 getRandomPointOnUnitSphere( curandStatePhilox4_32_10_t *rngState )
{
	double3 pointOnSphere = make_double3( 0., 0., 0. );
	
	double2 r1 = curand_uniform2_double ( &rngState[0] );
	double2 r2 = curand_uniform2_double ( &rngState[0] );
	
	pointOnSphere.x = (double) sqrt(-2.0*log(r1.x)) * sin(2*d_pi*r1.y);
	pointOnSphere.y = (double) sqrt(-2.0*log(r2.x)) * cos(2*d_pi*r2.y);
	pointOnSphere.z = (double) sqrt(-2.0*log(r2.x)) * sin(2*d_pi*r2.y);
	
	return pointOnSphere;
}

__device__ double3 selectAtomInDistribution( double dBdz, double Temp, curandStatePhilox4_32_10_t *rngState )
{
    double3 pos = make_double3( 0., 0., 0. );
    double3 r   = make_double3( 0., 0., 0. );

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

__device__ double3 getGaussianPoint( double mean, double std, curandStatePhilox4_32_10_t *rngState )
{
    double2 r1 = curand_normal2_double ( &rngState[0] ) * std + mean;
	double r2  = curand_normal_double  ( &rngState[0] ) * std + mean;
 
    double3 point = make_double3( r1.x, r1.y, r2 );
    
    return point;
}

__device__ bool pointIsInDistribution( double3 point, double dBdz, double Temp, curandStatePhilox4_32_10_t *rngState )
{
    bool pointIsIn = false;
    
    double potential   = 0.5*d_gs*d_muB*dBdz*sqrt( point.x*point.x + point.y*point.y + 4.*point.z*point.z );
    double probability = exp( -potential / d_kB / Temp );
    
    if ( curand_uniform_double ( &rngState[0] ) < probability ) {
        pointIsIn = true;
    }
    
    return pointIsIn;
}

__device__ double3 updateAccel( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    // The rsqrt function returns the reciprocal square root of its argument
	double potential = -0.5*d_gs*d_muB*d_dBdz*rsqrt(pos.x*pos.x + pos.y*pos.y + 4.0*pos.z*pos.z)/d_mRb;
	
	accel.x =       potential * pos.x;
	accel.y =       potential * pos.y;
	accel.z = 4.0 * potential * pos.z;
    
    return accel;
}

void initSigvrmax( double *d_sigvrmax, int numberOfCells )
{
    double sigvrmax = sqrt(3.*h_kB*Tinit/h_mRb)*8.*h_pi*h_a*h_a;
    
    cudaSetMem( d_sigvrmax, sigvrmax, numberOfCells + 1 );
}