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
                            hbool_t *d_isPerturb,
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
                                                 d_isPerturb,
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
                                    hbool_t *isPerturb,
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
        
        isPerturb[atom] = false;
        
        atomID[atom] = atom;
		
		// Copy state back to global memory
		rngState[atom] = localrngState;
        
        if (atom < 0.1*numberOfAtoms) {
            pos[atom] = 0.5 * pos[atom];
            vel[atom] = sqrt(0.5) * vel[atom];
            isPerturb[atom] = true;
        }
    }
    return;
}

__device__ double3 getRandomVelocity( double Temp, curandState_t *rngState )
{
	double3 vel = make_double3( 0., 0., 0. );
	
	double V = sqrt( d_kB*Temp/d_mRb);
	
	vel = V * getGaussianPoint( 0., 1., &rngState[0] );
    
	return vel;
}

__device__ double3 selectAtomInThermalDistribution( double Temp, curandState_t *rngState )
{
    double3 r   = make_double3( 0., 0., 0. );
    double3 pos = make_double3( 0., 0., 0. );
    
    bool noAtomSelected = true;
    while (noAtomSelected) {
        double2 r1 = curand_normal2_double ( &rngState[0] );
        double  r2 = curand_normal_double  ( &rngState[0] );
        
        double3 r = make_double3( r1.x, r1.y, r2 ) * d_maxGridWidth / 3;
        
        double U = -0.5*d_gs*d_muB*d_dBdz*sqrt(r.x*r.x+r.y*r.y+4.0*r.z*r.z);
        
        double Pr = exp( U / d_kB / Temp );
        
        if ( curand_uniform_double ( &rngState[0] ) < Pr) {
            pos = r;
            noAtomSelected = false;
        }
    }
    
    return pos;
}

__device__ double3 getGaussianPoint( double mean, double std, curandState_t *rngState )
{
    double2 r1 = curand_normal2_double ( &rngState[0] ) * std + mean;
	double  r2 = curand_normal_double  ( &rngState[0] ) * std + mean;
 
    double3 point = make_double3( r1.x, r1.y, r2 );
    
    return point;
}

__device__ double3 updateAccel( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    double potential = d_gs * d_muB * d_dBdz * rsqrt( pos.x*pos.x + pos.y*pos.y + 4.*pos.z*pos.z ) / d_mRb;
    
    accel.x =-0.5 * potential * pos.x;
    accel.y =-0.5 * potential * pos.y;
    accel.z =-2.0 * potential * pos.z;
    
    return accel;
}

void initSigvrmax( double *d_sigvrmax, int numberOfCells )
{
    double sigvrmax = sqrt(8.*h_kB*Tinit/(h_pi*h_mRb))*8.*h_pi*h_a*h_a;
    
    cudaSetMem( d_sigvrmax, sigvrmax, numberOfCells + 1 );
}