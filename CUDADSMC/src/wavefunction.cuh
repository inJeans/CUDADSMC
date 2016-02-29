//
//  wavefunction.h
//  CUDADSMC
//
//  Created by Christopher Watkins on 8/12/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef __CUDADSMC__wavefunction_cuh
#define __CUDADSMC__wavefunction_cuh

#include <cuda.h>
#include <curand_kernel.h>

#include "deviceSystemParameters.cuh"
#include <hdf5.h>
#include "math.h"
#include <cuComplex.h>
#include "vectorMath.cuh"

__global__ void evolveWavefunction(double3 *pos,
                                   cuDoubleComplex *psiUp,
                                   cuDoubleComplex *psiDn,
                                   int *atomID,
                                   int numberOfAtoms );

__device__ cuDoubleComplex updatePsiUp(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn );

__device__ cuDoubleComplex updatePsiDn(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn );

__global__ void getLocalPopulations(double3 *pos,
                                    cuDoubleComplex *psiUp,
                                    cuDoubleComplex *psiDn,
                                    double2 *localPopulations,
                                    int *atomID,
                                    int numberOfAtoms );

__global__ void flipAtoms(double3 *pos,
                          double3 *vel,
                          cuDoubleComplex *psiUp,
                          cuDoubleComplex *psiDn,
                          double2 *localPopulations,
                          hbool_t *atomIsSpinUp,
                          int *atomID,
                          curandState_t *rngState,
                          int numberOfAtoms );

__device__ double2 projectLocalPopulations(double3 pos,
                                           cuDoubleComplex psiUp,
                                           cuDoubleComplex psiDn );

__global__ void exponentialDecay(double3 *pos,
                                 cuDoubleComplex *psiUp,
                                 cuDoubleComplex *psiDn,
                                 int *atomID,
                                 double dt,
                                 int numberOfAtoms );

__device__ double calculateTau( double3 pos );

__device__ double3 getAcc( double3 pos );

__device__ double3 getMagFieldN( double3 pos );

__device__ double getAbsB( double3 pos );

__device__ double3 getBField( double3 pos );

__device__ double3 getBdx( double3 pos );
__device__ double3 getBdy( double3 pos );
__device__ double3 getBdz( double3 pos );

__global__ void normalise(cuDoubleComplex *psiUp,
                          cuDoubleComplex *psiDn,
                          int *atomID,
                          int numberOfAtoms );

#endif /* defined(__CUDADSMC__wavefunction_cuh) */
