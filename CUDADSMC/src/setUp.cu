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
int findRNGArrayLength( int numberOfCells, int numberOfAtoms )
{
    int sizeOfRNG = 0;
    
    if (numberOfAtoms > numberOfCells) {
		sizeOfRNG = numberOfAtoms;
	}
	else
	{
		sizeOfRNG = numberOfCells;
	}
    
    return sizeOfRNG;
}
void h_initRNG( curandState_t *d_rngStates, int sizeOfRNG )
{
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) initRNG,
                                        0,
                                        sizeOfRNG );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    int numSMs;
    cudaDeviceGetAttribute( &numSMs,
                            cudaDevAttrMultiProcessorCount,
                            device);
    
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    initRNG<<<gridSize,blockSize>>>( d_rngStates, sizeOfRNG );
    
    return;
}

__global__ void initRNG( curandState_t *rngState, int numberOfAtoms )
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

void h_generateInitialDist( double3 *d_pos,
                            double3 *d_vel,
                            double3 *d_acc,
                            int      numberOfAtoms,
                            double   Temp,
                            curandState_t *d_rngStates,
                            int *d_atomID )
{
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) generateInitialDist,
                                        0,
                                        sizeOfRNG );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    int numSMs;
    cudaDeviceGetAttribute( &numSMs,
                            cudaDevAttrMultiProcessorCount,
                            device);
    
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    generateInitialDist<<<gridSize,blockSize>>>( d_pos,
                                                 d_vel,
                                                 d_acc,
                                                 numberOfAtoms,
                                                 Tinit,
                                                 d_rngStates,
                                                 d_atomID );
    
    return;
}

// Kernel to generate the initial distribution
__global__ void generateInitialDist(double3 *pos,
                                    double3 *vel,
                                    double3 *acc,
                                    int      numberOfAtoms,
									double   Temp,
									curandState_t *rngState,
                                    int *atomID ) {
    
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		/* Copy state to local memory for efficiency */
		curandState_t localrngState = rngState[atom];
		
        pos[atom] = selectAtomInThermalDistribution( Temp,
                                                     &localrngState );
        
		vel[atom] = getRandomVelocity( Temp, &localrngState );
        
        acc[atom] = updateAccel( pos[atom] );
        
        atomID[atom] = atom;
		
		// Copy state back to global memory
		rngState[atom] = localrngState;
    }
    return;
}

__device__ double3 getRandomVelocity( double Temp, curandState_t *rngState )
{
	double3 vel = make_double3( 0., 0., 0. );
	
	double V = sqrt( d_kB*Temp/d_mRb);
	
	vel = getGaussianPoint( 0., V, &rngState[0] );
    
	return vel;
}

__device__ double3 selectAtomInThermalDistribution( double Temp, curandState_t *rngState )
{
    double r = sqrt( d_kB*Temp / (d_gs*d_muB*d_dBdr) );
        
    double3 pos = getGaussianPoint( 0., r, &rngState[0] );
    
    return pos;
}

__device__ double3 getGaussianPoint( double mean, double std, curandState_t *rngState )
{
    double r1 = curand_normal_double ( &rngState[0] ) * std + mean;
	double r2 = curand_normal_double ( &rngState[0] ) * std + mean;
    double r3 = curand_normal_double ( &rngState[0] ) * std + mean;
 
    double3 point = make_double3( r1, r2, r3 );
    
    return point;
}

__device__ double3 updateAccel( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    double potential = -1.0 * d_gs * d_muB * d_dBdr / d_mRb;
    
    accel = potential * pos;
    
    return accel;
}

void initSigvrmax( double *d_sigvrmax, int numberOfCells )
{
    double sigvrmax = sqrt(16.*h_kB*Tinit/(h_pi*h_mRb))*8.*h_pi*h_a*h_a;
    
    cudaSetMem( d_sigvrmax, sigvrmax, numberOfCells + 1 );
}