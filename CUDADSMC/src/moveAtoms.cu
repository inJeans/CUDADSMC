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
#include "math.h"

__global__ void copyConstantsToDevice( double dt )
{
	d_dt = dt;
	d_loopsPerCollision = 0.0007 / d_dt;
	
	return;
}

__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, int *atomID, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        double3 l_pos = pos[atomID[atom]];
        double3 l_vel = vel[atomID[atom]];
        double3 l_acc = acc[atomID[atom]];
        
        //        for (int i=0; i<d_loopsPerCollision; i++) {
        velocityVerletUpdate(&l_pos,
                             &l_vel,
                             &l_acc );
        //        }
        
        pos[atomID[atom]] = l_pos;
        vel[atomID[atom]] = l_vel;
        acc[atomID[atom]] = l_acc;
        
    }
    
    return;
}

__device__ void velocityVerletUpdate( double3 *pos, double3 *vel, double3 *acc )
{
    vel[0] = updateVelHalfStep( pos[0], vel[0], acc[0] );
    pos[0] = updatePos( pos[0], vel[0] );
    acc[0] = updateAcc( pos[0] );
    vel[0] = updateVelHalfStep( pos[0], vel[0], acc[0] );
    vel[0] = checkPos( pos[0], vel[0] );
    
    return;
}

__device__ void symplecticEulerUpdate( double3 *pos, double3 *vel, double3 *acc )
{
    acc[0] = updateAcc( pos[0] );
    vel[0] = updateVel( pos[0], vel[0], acc[0] );
    pos[0] = updatePos( pos[0], vel[0] );
    vel[0] = checkPos( pos[0], vel[0] );
}

__device__ double3 updateVel( double3 pos, double3 vel, double3 acc )
{
    return vel + acc * d_dt;
}

__device__ double3 updateVelHalfStep( double3 pos, double3 vel, double3 acc )
{
    double3 newVel = vel + 0.5 * acc * d_dt;
    
//    if (pos.x > d_maxGridWidth.x || pos.x < -d_maxGridWidth.x) {
//        newVel.x = -newVel.x;
//    }
//    
//    if (pos.y > d_maxGridWidth.y || pos.y < -d_maxGridWidth.y) {
//        newVel.y = -newVel.y;
//    }
//    
//    if (pos.z > d_maxGridWidth.z || pos.z < -d_maxGridWidth.z) {
//        newVel.z = -newVel.z;
//    }
    
    return newVel;
}

__device__ double3 updatePos( double3 pos, double3 vel )
{
    double3 newPos = pos + vel * d_dt;
    
//    if (newPos.x > d_maxGridWidth.x || newPos.x < -d_maxGridWidth.x) {
//        newPos.x = -newPos.x;
//    }
//    
//    if (newPos.y > d_maxGridWidth.y || newPos.y < -d_maxGridWidth.y) {
//        newPos.y = -newPos.y;
//    }
//    
//    if (newPos.z > d_maxGridWidth.z || newPos.z < -d_maxGridWidth.z) {
//        newPos.z = -newPos.z;
//    }
    
    return newPos;
}

__device__ double3 updateAcc( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    return accel;
}

__device__ double3 checkPos( double3 pos, double3 vel )
{
    
    if (pos.x > d_maxGridWidth.x || pos.x < -d_maxGridWidth.x) {
        vel.x = -vel.x;
    }
    
    if (pos.y > d_maxGridWidth.y || pos.y < -d_maxGridWidth.y) {
        vel.y = -vel.y;
    }
    
    if (pos.z > d_maxGridWidth.z || pos.z < -d_maxGridWidth.z) {
        vel.z = -vel.z;
    }
    
    return vel;
}