//
//  moveAtoms.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 12/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_moveAtoms_cuh
#define CUDADSMC_moveAtoms_cuh

__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, int numberOfAtoms );
__device__ double3 updateVelHalfStep( double3 vel, double3 acc );
__device__ double3 updatePos( double3 pos, double3 vel );
__device__ double3 updateAcc( double3 pos );

#endif
