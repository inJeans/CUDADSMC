//
//  spinEvolution.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 11/09/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <cuda.h>
#include <hdf5.h>
#include <curand_kernel.h>
#include <cuComplex.h>

#include "deviceSystemParameters.cuh"
#include "vectorMath.cuh"
#include "spinEvolution.cuh"
#include "moveAtoms.cuh"
#include "math.h"

// Kernel to evolve the spin of the atom using the
// unitary time evolution operator
__global__ void unitaryEvolution( zomplex *psiU, zomplex *psiD, double2 *oldPops2, double3 *pos, double3 *vel, int numberOfAtoms )
{
	for ( int atom = blockIdx.x * blockDim.x + threadIdx.x;
              atom < numberOfAtoms;
              atom += blockDim.x * gridDim.x )
    {
        d_dt = 1.0e-6;
//        d_loopsPerCollision = 0.0007 / d_dt;
        
		// Make a local copy of the position for increased
		// efficiency
		double3 l_pos = pos[atom];
        double3 l_vel = vel[atom];
		
        // These guys will be used often in the calculations below
		double3 B     = getMagneticField( l_pos + 0.5*l_vel*d_dt );
        double  magB  = length( B );
        double3 Bn    = B / magB;
		double  theta = 0.5*d_gs*d_muB*magB*d_dt / d_hbar;
		double  costheta = cos(theta);
		double  sintheta = sin(theta);
		
		// Unitary time evolution operator for the standard Hamiltonian
		zomplex U11 = make_cuDoubleComplex(        costheta, -Bn.z * sintheta );
		zomplex U12 = make_cuDoubleComplex(-Bn.y * sintheta, -Bn.x * sintheta );
		zomplex U21 = make_cuDoubleComplex( Bn.y * sintheta, -Bn.x * sintheta );
		zomplex U22 = make_cuDoubleComplex(        costheta,  Bn.z * sintheta );
        
		// Get local copies of the wavefunction for efficiency
		zomplex l_psiU = psiU[atom];
		zomplex l_psiD = psiD[atom];
		
		oldPops2[atom] = getEigenStatePopulations( l_psiD, l_psiU, Bn );
		
		// Write out the update values of the wavefunction
        // to global memory
		psiU[atom] = U11*l_psiU + U12*l_psiD;
		psiD[atom] = U21*l_psiU + U22*l_psiD;
		
	}
	
	return;
}

__device__ double3 getMagneticField( double3 pos )
{
    double3 magneticField = make_double3( 0., 0., 0. );
    
    magneticField.x = 0.5*d_dBdz*pos.x;
    magneticField.y = 0.5*d_dBdz*pos.y;
    magneticField.z =-1.0*d_dBdz*pos.z;
    
    return magneticField;
}

__device__ double3 getMagneticFieldNormal( double3 pos )
{
    double3 B     = getMagneticField( pos );
    double  magB  = length( B );
    double3 Bn    = B / magB;
    
    return Bn;
}

__device__ double2 getEigenStatePopulations( zomplex psiD, zomplex psiU, double3 Bn )
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

// This kernel will perform the spin projection
__global__ void projectSpins( zomplex *psiU, zomplex *psiD, double2 *oldPops2, double3 *pos, double3 *vel, hbool_t *isSpinUp, curandStatePhilox4_32_10_t *rngstate, int numberOfAtoms, double3 *flippedPos, double3 *flippedVel )
{
	for ( int atom = blockIdx.x * blockDim.x + threadIdx.x;
              atom < numberOfAtoms;
              atom += blockDim.x * gridDim.x )
    {
		zomplex l_psiU = psiU[atom];
		zomplex l_psiD = psiD[atom];
		
		double2 l_oldPops2 = oldPops2[atom];
		
		double3 l_pos = pos[atom];
		double3 l_vel = vel[atom];
        
        double3 B     = getMagneticField( l_pos );
        double  magB  = length( B );
        double3 Bn = getMagneticFieldNormal( l_pos );
		
        double2 newPops2 = getEigenStatePopulations( l_psiD, l_psiU, Bn );
//        if (atom==0) {
//            printf("x = (%g, %g, %g), Bn = (%g, %g, %g)\n", l_pos.x, l_pos.y, l_pos.z, Bn.x, Bn.y, Bn.z );
//        }
        // Copy rng state to local memory for efficiency
		curandStatePhilox4_32_10_t l_rngstate = rngstate[atom];
        
		if (isSpinUp[atom]) {
            
            double pFlip = ( newPops2.y - l_oldPops2.y ) / l_oldPops2.x;
            
            if ( curand_uniform_double (&l_rngstate) < pFlip ) {
				isSpinUp[atom] = !isSpinUp[atom];
                flippedPos[atom] = l_pos;
                flippedVel[atom] = l_vel;
                
				double deltaE = d_gs*d_muB*d_dBdz*magB;
                double Ek = 0.5 * d_mRb * ( l_vel.x*l_vel.x + l_vel.y*l_vel.y + l_vel.z*l_vel.z );
                
				l_vel = sqrt( Ek + 2.*deltaE ) * l_vel / length( l_vel );
//                printf( " I flipped? Pflip = %g\n", pFlip );
			}
		}
		else {
			double deltaE = d_gs * d_muB * d_dBdz * magB;
			double Ek = 0.5 * d_mRb * ( l_vel.x*l_vel.x + l_vel.y*l_vel.y + l_vel.z*l_vel.z );
            
			if ( deltaE < Ek )
            {
                double pFlip = ( newPops2.x - l_oldPops2.x ) / l_oldPops2.y;
                
                if ( curand_uniform_double (&l_rngstate) < pFlip ) {
					isSpinUp[atom] = !isSpinUp[atom];
                    
                    l_vel = sqrt( Ek - 2.*deltaE ) * l_vel / length( l_vel );
				}
			}
		}
		
		// Copy state back to global memory
		rngstate[atom] = l_rngstate;
        vel[atom] = l_vel;
	}
	return;
}

__global__ void exponentialDecay( zomplex *psiU, zomplex *psiD, double3 *pos, hbool_t *isSpinUp, int numberOfAtoms )
{
	for ( int atom = blockIdx.x * blockDim.x + threadIdx.x;
              atom < numberOfAtoms;
              atom += blockDim.x * gridDim.x )
    {
        d_dt = 1.0e-6;
//        d_loopsPerCollision = 0.0007 / d_dt;
        
		double3 l_pos = pos[atom];
		
		double3 Bn = getMagneticFieldNormal( l_pos );
		
		double delta_a = length( 2. * updateTheAcc( l_pos ) );
		double tau = 1.163*sqrt( d_wavelength / delta_a );
		
        // This guy appears a few times
		double expx  = exp( -d_dt / tau );
		
		// Elements of the exponential decay matrix
		zomplex D11, D12, D21, D22;
		
		if (isSpinUp[atom]) {
			D11 = 0.5 * make_cuDoubleComplex( 1. - expx*(Bn.z-1.) + Bn.z, 0. );
			D12 = 0.5 * make_cuDoubleComplex( ( 1. - expx ) * Bn.x, (-1. + expx ) * Bn.y );
			D21 = 0.5 * make_cuDoubleComplex( ( 1. - expx ) * Bn.x, (-1. + expx ) * Bn.y );
			D22 = 0.5 * make_cuDoubleComplex( 1. + expx*(Bn.z+1.) - Bn.z, 0. );
		}
		else {
			D11 = 0.5 * make_cuDoubleComplex( 1. + expx*(Bn.z+1.) - Bn.z, 0. );
			D12 = 0.5 * make_cuDoubleComplex( (-1. + expx ) * Bn.x, ( 1. - expx ) * Bn.y );
			D21 = 0.5 * make_cuDoubleComplex( (-1. + expx ) * Bn.x, ( 1. - expx ) * Bn.y );
			D22 = 0.5 * make_cuDoubleComplex( 1. - expx*(Bn.z-1.) + Bn.z, 0. );
		}
		
		// Get local copies of the wavefunction for efficiency
		zomplex l_psiU = psiU[atom];
		zomplex l_psiD = psiD[atom];
		
		// Write out the update values of the wavefunction
		// to global memory
		psiU[atom] = D11*l_psiU + D12*l_psiD;
		psiD[atom] = D21*l_psiU + D22*l_psiD;
        
	}
	
	return;
}

__global__ void normaliseWavefunction( zomplex *psiU, zomplex *psiD, int numberOfAtoms )
{
	for ( int atom = blockIdx.x * blockDim.x + threadIdx.x;
              atom < numberOfAtoms;
              atom += blockDim.x * gridDim.x)
    {
		// Get local copies of the wavefunction for efficiency
		zomplex l_psiU = psiU[atom];
		zomplex l_psiD = psiD[atom];
		
		double N2 = l_psiU.x*l_psiU.x + l_psiU.y*l_psiU.y + l_psiD.x*l_psiD.x + l_psiD.y*l_psiD.y;
		
		// Write out the update values of the wavefunction
		// to global memory
		psiU[atom] = l_psiU * rsqrt( N2 );
		psiD[atom] = l_psiD * rsqrt( N2 );
	}
	
	return;
}

__device__ double3 updateTheAcc( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    // The rsqrt function returns the reciprocal square root of its argument
	double potential = -0.5*d_gs*d_muB*d_dBdz*rsqrt(pos.x*pos.x + pos.y*pos.y + 4.0*pos.z*pos.z)/d_mRb;
	
	accel.x =       potential * pos.x;
	accel.y =       potential * pos.y;
	accel.z = 4.0 * potential * pos.z;
    
    return accel;
}
