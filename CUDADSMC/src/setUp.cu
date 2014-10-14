//
//  setUp.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include "vectorMath.cuh"
#include "setUp.cuh"
#include "math.h"
#include "cudaHelpers.cuh"

#include "declareInitialSystemParameters.cuh"
#include "deviceSystemParameters.cuh"

#pragma mark - Random Number Generator
int findRNGArrayLength( void )
{
    int sizeOfRNG = 0;
    
    if (numberOfAtoms > 64*numberOfCells) {
		sizeOfRNG = numberOfAtoms;
	}
	else
	{
		sizeOfRNG = 64*numberOfCells;
	}
    
    return sizeOfRNG;
}

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
									curandStatePhilox4_32_10_t *rngState) {
    
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		/* Copy state to local memory for efficiency */
		curandStatePhilox4_32_10_t localrngState = rngState[atom];
		
        pos[atom] = selectAtomInBox( &localrngState );
        
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
	
	double V = sqrt( d_kB*Temp/d_mRb);
	
	vel = V * getGaussianPoint( 0., 1., &rngState[0] );
    
	return vel;
}

__device__ double3 selectAtomInBox( curandStatePhilox4_32_10_t *rngState )
{
    double3 r   = make_double3( 0., 0., 0. );
    
    double2 r1 = ( curand_uniform2_double ( &rngState[0] ) * 2. - 1. );
    double  r2 = ( curand_uniform_double  ( &rngState[0] ) * 2. - 1. );
    
    double3 pos = make_double3( r1.x, r1.y, r2 ) * d_maxGridWidth;
    
    return pos;
}

__device__ double3 getGaussianPoint( double mean, double std, curandStatePhilox4_32_10_t *rngState )
{
    double2 r1 = curand_normal2_double ( &rngState[0] ) * std + mean;
	double  r2 = curand_normal_double  ( &rngState[0] ) * std + mean;
 
    double3 point = make_double3( r1.x, r1.y, r2 );
    
    return point;
}

__device__ double3 updateAccel( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    return accel;
}

void initSigvrmax( double *d_sigvrmax, int numberOfCells )
{
    double sigvrmax = sqrt(8.*h_kB*Tinit/(h_pi*h_mRb))*8.*h_pi*h_a*h_a;
    
    cudaSetMem( d_sigvrmax, sigvrmax, numberOfCells + 1 );
}