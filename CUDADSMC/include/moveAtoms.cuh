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
#include "math.h"
#include <cuComplex.h>

__global__ void copyConstantsToDevice( double dt );
__global__ void moveAtoms( double3 *pos, double3 *vel, double3 *acc, cuDoubleComplex *psiUp, cuDoubleComplex *psiDn, int *atomID, int numberOfAtoms );
__device__ void velocityVerletUpdate( double3 *pos, double3 *vel, double3 *acc, cuDoubleComplex *psiUp, cuDoubleComplex *psiDn );
__device__ void symplecticEulerUpdate( double3 *pos, double3 *vel, double3 *acc, cuDoubleComplex *psiUp, cuDoubleComplex *psiDn );
__device__ double3 updateVel( double3 pos, double3 vel, double3 acc );
__device__ double3 updateVelHalfStep( double3 pos, double3 vel, double3 acc );
__device__ double3 updatePos( double3 pos, double3 vel );
__device__ double3 updateAcc( cuDoubleComplex psiUp, cuDoubleComplex psiDn );
__device__ cuDoubleComplex updatePsiUp(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn );
__device__ cuDoubleComplex updatePsiDn(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn );
__device__ double3 getMagneticFieldNormal( double3 pos );
__device__ double  getMagB( double3 pos );
__device__ double3 getMagneticField( double3 pos );

#endif
