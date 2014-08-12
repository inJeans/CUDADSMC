//
//  moveAtoms.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 31/07/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <cuda.h>

#include "moveAtoms.cuh"
#include "vectorMath.cuh"
#include "math.h"

////////////////////////////////////////////////////////////////////////////////
// Define some global variables on the device                                 //
////////////////////////////////////////////////////////////////////////////////

__constant__ double d_gs   =  0.5;				// Gyromagnetic ratio
__constant__ double d_MF   = -1.0;				// Magnetic quantum number
__constant__ double d_muB  = 9.27400915e-24;	// Bohr magneton
__constant__ double d_mRb  = 1.443160648e-25;	// 87Rb mass
__constant__ double d_pi   = 3.14159265;		// Pi
__constant__ double d_a    = 5.3e-9;			// Constant cross-section formula
__constant__ double d_kB   = 1.3806503e-23;		// Boltzmann's Constant
__constant__ double d_hbar = 1.05457148e-34;	// hbar

__constant__ double d_dBdz = 2.5;	    // field gradient
__constant__ double d_dt   = 1.0e-6;	// time step
__device__   int    loopsPerCollision = 100;	// loops per collision

/////////////////////////////////////////////////////////////////////////////////

__global__ void moveAtoms( double4 *pos, double4 *vel, double4 *acc, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
		double4 l_pos = pos[atom];
        double4 l_vel = vel[atom];
        double4 l_acc = acc[atom];
        
        for (int i=0; i<loopsPerCollision; loopsPerCollision++) {
            l_vel = updateVelHalfStep( l_vel, l_acc );
            l_pos = updatePos( l_pos, l_vel );
            l_acc = updateAcc( l_pos );
            l_vel = updateVelHalfStep( l_vel, l_acc );
        }
        
        pos[atom] = l_pos;
        vel[atom] = l_vel;
        acc[atom] = l_acc;
		
    }
    
    return;
}

__device__ double4 updateVelHalfStep( double4 vel, double4 acc )
{
    return vel + 0.5 * acc * d_dt;
}

__device__ double4 updatePos( double4 pos, double4 vel )
{
    return pos + pos * d_dt;
}

__device__ double4 updateAcc( double4 pos )
{
    double4 accel = make_double4( 0., 0., 0., 0. );
    
    // The rsqrt function returns the reciprocal square root of its argument
	double potential = -0.5*d_gs*d_muB*d_dBdz*rsqrt(pos.x*pos.x + pos.y*pos.y + 4.0*pos.z*pos.z)/d_mRb;
	
	accel.x =       potential * pos.x;
	accel.y =       potential * pos.y;
	accel.z = 4.0 * potential * pos.z;
    
    return accel;
}