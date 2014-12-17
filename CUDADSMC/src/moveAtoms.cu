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
	
	return;
}

__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, hbool_t *atomIsSpinUp, int *atomID, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        int l_atom = atomID[atom];
		double3 l_pos = pos[l_atom];
        double3 l_vel = vel[l_atom];
        double3 l_acc = acc[l_atom];
        hbool_t l_atomIsSpinUp = atomIsSpinUp[l_atom];
        
//        for (int i=0; i<d_loopsPerCollision; i++) {
            velocityVerletUpdate(&l_pos,
                                 &l_vel,
                                 &l_acc,
                                 l_atomIsSpinUp );
//        }
    
        pos[l_atom] = l_pos;
        vel[l_atom] = l_vel;
        acc[l_atom] = l_acc;
		
    }
    
    return;
}

__device__ void velocityVerletUpdate( double3 *pos, double3 *vel, double3 *acc, hbool_t atomIsSpinUp )
{
    vel[0]   = updateVelHalfStep(pos[0],
                                 vel[0],
                                 acc[0] );
    pos[0]   = updatePos(pos[0],
                         vel[0] );
    acc[0]   = updateAcc(pos[0],
                         atomIsSpinUp );
    vel[0]   = updateVelHalfStep(pos[0],
                                 vel[0],
                                 acc[0] );
    
    return;
}

__device__ void symplecticEulerUpdate( double3 *pos, double3 *vel, double3 *acc, hbool_t atomIsSpinUp )
{
    acc[0] = updateAcc(pos[0],
                       atomIsSpinUp );
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
    return pos + vel * d_dt;
}

__device__ double3 updateAcc( double3 pos, hbool_t atomIsSpinUp )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    double3 Bn = getMagneticFieldNormal( pos );
    double3 dBdx = getBdiffx( pos );
    double3 dBdy = getBdiffy( pos );
    double3 dBdz = getBdiffz( pos );
    
    double potential = -0.5 * d_gs * d_muB / d_mRb;
    
    if (atomIsSpinUp) {
        accel.x = potential * dot( dBdx, Bn );
        accel.y = potential * dot( dBdy, Bn );
        accel.z = potential * dot( dBdz, Bn );
    }
    else {
        accel.x =-1. * potential * dot( dBdx, Bn );
        accel.y =-1. * potential * dot( dBdy, Bn );
        accel.z =-1. * potential * dot( dBdz, Bn );
    }

    return accel;
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
    double3 B = make_double3( d_Bt, 0.0,-d_dBdz * pos.z );
    
    return B;
}

__device__ double3 getBdiffx( double3 pos )
{
    return make_double3( 0.0, 0.0, 0.0 );
}

__device__ double3 getBdiffy( double3 pos )
{
    return make_double3( 0.0, 0.0, 0.0 );
}

__device__ double3 getBdiffz( double3 pos )
{
    return make_double3( 0.0, 0.0,-d_dBdz );
}