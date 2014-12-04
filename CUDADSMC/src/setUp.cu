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
                           cuDoubleComplex *d_psiUp,
                           cuDoubleComplex *d_psiDn,
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
                                                 d_psiUp,
                                                 d_psiDn,
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
                                    cuDoubleComplex *psiUp,
                                    cuDoubleComplex *psiDn,
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
        
        psiUp[atom] = getAlignedSpinUp( pos[atom] );
        psiDn[atom] = getAlignedSpinDn( pos[atom] );
        
        acc[atom] = updateAccel( pos[atom],
                                 psiUp[atom],
                                 psiDn[atom]);
        
        atomID[atom] = atom;
		
		// Copy state back to global memory
		rngState[atom] = localrngState;
    }
    return;
}

__device__ double3 getRandomVelocity( double Temp, curandState_t *rngState )
{
	double3 vel = make_double3( 0., 0., 0. );
	
//	double V = sqrt( d_kB*Temp/d_mRb);
	
//	vel = getGaussianPoint( 0., V, &rngState[0] );
    
    double V = sqrt( 3.*d_kB*Temp/d_mRb);
    vel = V * getRndPointOnSphere( &rngState[0] );
    
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
        
        double3 r = make_double3( r1.x, r1.y, r2 ) * d_maxGridWidth / 10.;
        
        double3 B = getMagField( r );
        double  magB = length( B );
        double3 Bn = getMagFieldNormal( r );
        
        
        
        double U = 0.5 * (magB - d_B0) * d_gs * d_muB;
        
        double Pr = exp(-U / d_kB / Temp);
        
        if ( curand_uniform_double ( &rngState[0] ) < Pr) {
            pos = r;
            noAtomSelected = false;
        }
    }
    
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

__device__ double3 getRndPointOnSphere( curandState_t *rngState )
{
    double r1 = curand_normal_double ( rngState );
    double r2 = curand_normal_double ( rngState );
    double r3 = curand_normal_double ( rngState );
    
    double3 pointOnSphere = make_double3( r1, r2, r3 ) * rsqrt( r1*r1 + r2*r2 + r3*r3 );
    
    return pointOnSphere;
}

__device__ double3 updateAccel( double3 pos, cuDoubleComplex psiUp, cuDoubleComplex psiDn )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    double3 dBdx = diffMagFieldAlongx( pos );
    double3 dBdy = diffMagFieldAlongy( pos );
    double3 dBdz = diffMagFieldAlongz( pos );
    
    double potential = -1.0 * d_gs * d_muB / d_mRb;
    
    accel.x = potential * ( dBdx.x * (psiUp.x*psiDn.x + psiUp.y*psiDn.y) +
                            dBdx.y * (psiUp.x*psiDn.y - psiUp.y*psiDn.x) +
                            dBdx.z * (psiUp.x*psiUp.x + psiUp.y*psiUp.y - 0.5) );
    accel.y = potential * ( dBdy.x * (psiUp.x*psiDn.x + psiUp.y*psiDn.y) +
                            dBdy.y * (psiUp.x*psiDn.y - psiUp.y*psiDn.x) +
                            dBdy.z * (psiUp.x*psiUp.x + psiUp.y*psiUp.y - 0.5) );
//    accel.z = potential * ( dBdz.x * (psiUp.x*psiDn.x + psiUp.y*psiDn.y) +
//                            dBdz.y * (psiUp.x*psiDn.y - psiUp.y*psiDn.x) +
//                            dBdz.z * (psiUp.x*psiUp.x + psiUp.y*psiUp.y - 0.5) );
    accel.z = 0.5*potential * d_d2Bdx2 * pos.z;
    
    return accel;
}

__device__ cuDoubleComplex getAlignedSpinUp( double3 pos )
{
    double3 Bn = getMagFieldNormal( pos );
    
    cuDoubleComplex psiUp = 0.5 * make_cuDoubleComplex( 1. + Bn.x + Bn.z, -Bn.y ) * rsqrt( 1 + Bn.x );
    
    return psiUp;
}

__device__ cuDoubleComplex getAlignedSpinDn( double3 pos )
{
    double3 Bn = getMagFieldNormal( pos );
    
    cuDoubleComplex psiDn = 0.5 * make_cuDoubleComplex( 1. + Bn.x - Bn.z,  Bn.y ) * rsqrt( 1 + Bn.x );
    
    return psiDn;
}

__device__ double3 getMagFieldNormal( double3 pos )
{
    double3 B = getMagField( pos );
    
    double3 Bn = B / length( B );
    
    return Bn;
}

__device__ double3 getMagField( double3 pos )
{
    double3 B = d_B0     * make_double3( 0., 0., 1. ) +
                d_dBdx   * make_double3( pos.x, -pos.y, 0. ) +
          0.5 * d_d2Bdx2 * make_double3( -pos.x*pos.z, -pos.y*pos.z, pos.z*pos.z - 0.5*(pos.x*pos.x+pos.y*pos.y) );
    
    return B;
}

__device__ double3 diffMagFieldAlongx( double3 pos )
{
    double3 dBdx = make_double3( d_dBdx - 0.5 * d_d2Bdx2 * pos.z,
                                 0.,
                                -0.5 * d_d2Bdx2 * pos.x );
    return dBdx;
}

__device__ double3 diffMagFieldAlongy( double3 pos )
{
    double3 dBdy = make_double3( 0.,
                                -d_dBdx - 0.5 * d_d2Bdx2 * pos.z,
                                -0.5 * d_d2Bdx2 * pos.y );
    return dBdy;
}

__device__ double3 diffMagFieldAlongz( double3 pos )
{
    double3 dBdz = make_double3(-0.5 * d_d2Bdx2 * pos.x,
                                -0.5 * d_d2Bdx2 * pos.y,
                                 d_d2Bdx2 * pos.z );
    return dBdz;
}

void initSigvrmax( double *d_sigvrmax, int numberOfCells )
{
    double sigvrmax = sqrt(16.*h_kB*Tinit/(h_pi*h_mRb))*8.*h_pi*h_a*h_a;
    
    cudaSetMem( d_sigvrmax, sigvrmax, numberOfCells + 1 );
}