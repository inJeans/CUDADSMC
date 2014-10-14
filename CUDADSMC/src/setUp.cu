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
                                    hbool_t *isSpinUp,
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
        
        isSpinUp[atom] = true;
		
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

__device__ double3 selectAtomInDistribution( double dBdz, double Temp, curandStatePhilox4_32_10_t *rngState )
{
    double3 r   = make_double3( 0., 0., 0. );

    double max = 0.0001;
    
    double2 r1 = ( curand_uniform2_double ( &rngState[0] ) * 2. - 1. ) * max;
    double  r2 = ( curand_uniform_double  ( &rngState[0] ) * 2. - 1. ) * max;
    
    double3 pos = make_double3( r1.x, r1.y, r2 );
    
    return pos;
}

__device__ double3 getGaussianPoint( double mean, double std, curandStatePhilox4_32_10_t *rngState )
{
    double2 r1 = curand_normal2_double ( &rngState[0] ) * std + mean;
	double  r2 = curand_normal_double  ( &rngState[0] ) * std + mean;
 
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

void setInitialWavefunction( zomplex *d_psiU, zomplex *d_psiD, double2 *d_oldPops2, hbool_t *d_isSpinUp, double3 *d_pos, int numberOfAtoms )
{
    int blockSize;
	int minGridSize;
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) deviceSetInitialWavefunction,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    
    deviceSetInitialWavefunction<<<gridSize,blockSize>>>( d_psiU,
                                                          d_psiD,
                                                          d_oldPops2,
                                                          d_isSpinUp,
                                                          d_pos,
                                                          numberOfAtoms );
    
    return;
}

__global__ void deviceSetInitialWavefunction( zomplex *psiU, zomplex *psiD, double2 *oldPops2, hbool_t *isSpinUp, double3 *pos, int numberOfAtoms )
{
    for ( int atom = blockIdx.x * blockDim.x + threadIdx.x;
              atom < numberOfAtoms;
              atom += blockDim.x * gridDim.x )
    {
        double3 l_pos = pos[atom];
		double3 Bn = magneticFieldNormal( l_pos );
		
		zomplex l_psiU = 0.5 * make_cuDoubleComplex ( 1.+Bn.x+Bn.z, -Bn.y ) * rsqrt(1.+Bn.x);
		zomplex l_psiD = 0.5 * make_cuDoubleComplex ( 1.+Bn.x-Bn.z, +Bn.y ) * rsqrt(1.+Bn.x);
        
		isSpinUp[atom] = true;
		
		oldPops2[atom] = getEigenStatePops( l_psiD,
                                            l_psiU,
                                            Bn );
        
//        if (atom==0) {
//            printf("x = (%g, %g, %g), Bn = (%g, %g, %g)\n", l_pos.x, l_pos.y, l_pos.z, Bn.x, Bn.y, Bn.z );
//        }
        
        psiU[atom] = l_psiU;
        psiD[atom] = l_psiD;
	}
    
    return;
}

void initSigvrmax( double *d_sigvrmax, int numberOfCells )
{
    double sigvrmax = sqrt(3.*h_kB*Tinit/h_mRb)*8.*h_pi*h_a*h_a;
    
    cudaSetMem( d_sigvrmax, sigvrmax, numberOfCells + 1 );
}

__device__ double3 magneticField( double3 pos )
{
    double3 magneticField = make_double3( 0., 0., 0. );
    
    magneticField.x = 0.5*d_dBdz*pos.x;
    magneticField.y = 0.5*d_dBdz*pos.y;
    magneticField.z =-1.0*d_dBdz*pos.z;
    
    return magneticField;
}

__device__ double3 magneticFieldNormal( double3 pos )
{
    double3 B     = magneticField( pos );
    double  magB  = length( B );
    double3 Bn    = B / magB;
    
    return Bn;
}

__device__ double2 getEigenStatePops( zomplex psiD, zomplex psiU, double3 Bn )
{
    double2 statePopulations = make_double2( 0., 0. );
    
    // Record old populations
    statePopulations.x = 0.5 + Bn.x * ( psiD.x*psiU.x + psiD.y*psiU.y )
                             + Bn.y * ( psiD.y*psiU.x - psiD.x*psiU.y )
                             + Bn.z * ( psiU.x*psiU.x + psiU.y*psiU.y - 0.5 );
    statePopulations.y = 0.5 - Bn.x * ( psiD.x*psiU.x + psiD.y*psiU.y )
                             + Bn.y * (-psiD.y*psiU.x + psiD.x*psiU.y )
                             + Bn.z * (-psiU.x*psiU.x - psiU.y*psiU.y + 0.5 );
    
    return statePopulations;
}