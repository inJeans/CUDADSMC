//
//  moveAtoms.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 12/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_moveAtoms_cuh
#define CUDADSMC_moveAtoms_cuh

__global__ void moveAtoms( double4 *pos, double4 *vel, double4 *acc, int numberOfAtoms );
__device__ double4 updateVelHalfStep( double4 vel, double4 acc );
__device__ double4 updatePos( double4 pos, double4 vel );
__device__ double4 updateAcc( double4 pos );

#endif
