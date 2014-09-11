//
//  moveAtoms.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 31/07/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <cuda.h>

#include "declareDeviceSystemParameters.cuh"
#include "moveAtoms.cuh"
#include "vectorMath.cuh"
#include "math.h"

__global__ void copyConstantsToDevice( double dt )
{
	d_dt = dt;
	d_loopsPerCollision = 0.01 / d_dt;
	
	return;
}

__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		double3 l_pos = pos[atom];
        double3 l_vel = vel[atom];
        double3 l_acc = acc[atom];
		
        for (int i=0; i<d_loopsPerCollision; i++) {
            velocityVerletUpdate( &l_pos,
                                  &l_vel,
                                  &l_acc );
        }
    
        pos[atom] = l_pos;
        vel[atom] = l_vel;
        acc[atom] = l_acc;
		
    }
    
    return;
}

__device__ void velocityVerletUpdate( double3 *pos, double3 *vel, double3 *acc )
{
    vel[0] = updateVelHalfStep( vel[0], acc[0] );
    pos[0] = updatePos( pos[0], vel[0] );
    acc[0] = updateAcc( pos[0] );
    vel[0] = updateVelHalfStep( vel[0], acc[0] );
}

__device__ void symplecticEulerUpdate( double3 *pos, double3 *vel, double3 *acc )
{
    acc[0] = updateAcc( pos[0] );
    vel[0] = updateVel( vel[0], acc[0] );
    pos[0] = updatePos( pos[0], vel[0] );
}

__device__ double3 updateVel( double3 vel, double3 acc )
{
    return vel + acc * d_dt;
}

__device__ double3 updateVelHalfStep( double3 vel, double3 acc )
{
    return vel + 0.5 * acc * d_dt;
}

__device__ double3 updatePos( double3 pos, double3 vel )
{
    return pos + vel * d_dt;
}

__device__ double3 updateAcc( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    // The rsqrt function returns the reciprocal square root of its argument
	double potential = -0.5*d_gs*d_muB*d_dBdz*rsqrt(pos.x*pos.x + pos.y*pos.y + 4.0*pos.z*pos.z)/d_mRb;
	
	accel.x =       potential * pos.x;
	accel.y =       potential * pos.y;
	accel.z = 4.0 * potential * pos.z;
    
    return accel;
}