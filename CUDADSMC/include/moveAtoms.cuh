//
//  moveAtoms.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 12/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_moveAtoms_cuh
#define CUDADSMC_moveAtoms_cuh

#include <hdf5.h>

__global__ void copyConstantsToDevice( double dt );
__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, int *atomID, int numberOfAtoms );
__device__ void velocityVerletUpdate( double3 *pos, double3 *vel, double3 *acc );
__device__ void symplecticEulerUpdate( double3 *pos, double3 *vel, double3 *acc );
__device__ void explicitEulerUpdate( double3 *pos, double3 *vel, double3 *acc );
__device__ double3 updateVel( double3 pos, double3 vel, double3 acc );
__device__ double3 updateVelHalfStep( double3 pos, double3 vel, double3 acc );
__device__ double3 updatePos( double3 pos, double3 vel );
__device__ double3 updateAcc( double3 pos );

#endif
