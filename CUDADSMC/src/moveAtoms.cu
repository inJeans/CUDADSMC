//
//  moveAtoms.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 31/07/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <cuda.h>

#include "deviceSystemParameters.cuh"
#include "moveAtoms.cuh"
#include "vectorMath.cuh"

__global__ void copyConstantsToDevice( double dt )
{
	d_dt = dt;
	d_loopsPerCollision = 0.0007 / d_dt;
	
	return;
}

__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, cuDoubleComplex *psiUp, cuDoubleComplex *psiDn, int *atomID, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        int l_atom = atomID[atom];
		double3 l_pos = pos[l_atom];
        double3 l_vel = vel[l_atom];
        double3 l_acc = acc[l_atom];
        cuDoubleComplex l_psiUp = psiUp[l_atom];
        cuDoubleComplex l_psiDn = psiDn[l_atom];
		
//        for (int i=0; i<d_loopsPerCollision; i++) {
            velocityVerletUpdate(&l_pos,
                                  &l_vel,
                                  &l_acc,
                                  &l_psiUp,
                                  &l_psiDn);
//        }
    
        pos[l_atom] = l_pos;
        vel[l_atom] = l_vel;
        acc[l_atom] = l_acc;
        psiUp[l_atom] = l_psiUp;
        psiDn[l_atom] = l_psiDn;
		
    }
    
    return;
}

__device__ void velocityVerletUpdate( double3 *pos, double3 *vel, double3 *acc, cuDoubleComplex *psiUp, cuDoubleComplex *psiDn )
{
    cuDoubleComplex psiUpTemp = psiUp[0];
    psiUp[0] = updatePsiUp(pos[0],
                           psiUp[0],
                           psiDn[0] );
    psiDn[0] = updatePsiDn(pos[0],
                           psiUpTemp,
                           psiDn[0] );
    
    vel[0]   = updateVelHalfStep(pos[0],
                                 vel[0],
                                 acc[0] );
    pos[0]   = updatePos(pos[0],
                         vel[0] );
    acc[0]   = updateAcc(psiUp[0],
                         psiDn[0] );
    vel[0]   = updateVelHalfStep(pos[0],
                                 vel[0],
                                 acc[0] );
    
    return;
}

__device__ void symplecticEulerUpdate( double3 *pos, double3 *vel, double3 *acc, cuDoubleComplex *psiUp, cuDoubleComplex *psiDn )
{
    cuDoubleComplex psiUpTemp = psiUp[0];
    psiUp[0] = updatePsiUp(pos[0],
                           psiUp[0],
                           psiDn[0] );
    psiDn[0] = updatePsiDn(pos[0],
                           psiUpTemp,
                           psiDn[0] );
    
    acc[0] = updateAcc(psiUp[0],
                       psiDn[0] );
    vel[0] = updateVel(pos[0],
                       vel[0],
                       acc[0] );
    pos[0]   = updatePos(pos[0],
                         vel[0] );
}

__device__ double3 updateVel( double3 pos, double3 vel, double3 acc )
{
    return vel + acc * d_dt;
}

__device__ double3 updateVelHalfStep( double3 pos, double3 vel, double3 acc )
{
    return vel + 0.5 * acc * d_dt;
}

__device__ double3 updatePos( double3 pos, double3 vel )
{
    double3 newPos = pos + vel * d_dt;
    
    return newPos;
}

__device__ double3 updateAcc( cuDoubleComplex psiUp, cuDoubleComplex psiDn )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    double potential = -1.0 * d_gs * d_muB * d_dBdz / d_mRb;
    
    accel.x = potential * (psiUp.x*psiDn.x + psiUp.y*psiDn.y);
    accel.y = potential * (psiUp.x*psiDn.y - psiUp.y*psiDn.x);
    accel.z = potential * (1. - 2.*psiUp.x*psiUp.x - 2.*psiUp.y*psiUp.y);
    
    
    return accel;
}

__device__ cuDoubleComplex updatePsiUp(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn )
{
    double3 Bn = getMagneticFieldNormal( pos );
    double  B  = getMagB( pos );
    
    double theta = 0.5 * d_gs * d_muB * B * d_dt / d_hbar;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    
    cuDoubleComplex newPsiUp = make_cuDoubleComplex( psiUp.x*cosTheta + ( Bn.x*psiDn.y - Bn.y*psiDn.x + Bn.z*psiUp.y)*sinTheta,
                                                     psiUp.y*cosTheta + (-Bn.x*psiDn.x - Bn.y*psiDn.y - Bn.z*psiUp.x)*sinTheta );
    
//    cuDoubleComplex newPsiUp = 0.5 * make_cuDoubleComplex( 1. + Bn.x + Bn.z, -Bn.y ) * rsqrt( 1 + Bn.x );
    
    return newPsiUp;
}

__device__ cuDoubleComplex updatePsiDn(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn )
{
    double3 Bn = getMagneticFieldNormal( pos );
    double  B  = getMagB( pos );
    
    double theta = 0.5 * d_gs * d_muB * B * d_dt / d_hbar;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    
    cuDoubleComplex newPsiDn = make_cuDoubleComplex( psiDn.x*cosTheta + ( Bn.x*psiUp.y + Bn.y*psiUp.x - Bn.z*psiDn.y)*sinTheta,
                                                     psiDn.y*cosTheta + (-Bn.x*psiUp.x + Bn.y*psiUp.y + Bn.z*psiDn.x)*sinTheta );
    
//    cuDoubleComplex newPsiDn = 0.5 * make_cuDoubleComplex( 1. + Bn.x - Bn.z,  Bn.y ) * rsqrt( 1 + Bn.x );
    
    return newPsiDn;
}

__device__ double3 getMagneticFieldNormal( double3 pos )
{
    double3 B = getMagneticField( pos );
    
    double3 Bn = B / length( B );
    
    return Bn;
}

__device__ double getMagB( double3 pos )
{
    double3 B = getMagneticField( pos );
    
    return length( B );
}

__device__ double3 getMagneticField( double3 pos )
{
    double3 B = d_dBdz * make_double3( 0.5 * pos.x, 0.5 * pos.y, -pos.z );
    
    return B;
}