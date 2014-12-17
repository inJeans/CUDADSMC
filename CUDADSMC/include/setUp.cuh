//
//  setUp.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_setUp_cuh
#define CUDADSMC_setUp_cuh

#include <stdio.h>
#include <cuda.h>
#include "math.h"
#include <cuComplex.h>
#include <curand_kernel.h>
#include <hdf5.h>

int findRNGArrayLength( int numberOfCells, int numberOfAtoms );

void h_initRNG( curandState_t *rngState,
                int numberOfAtoms );

__global__ void initRNG( curandState_t *rngState,
                         int numberOfAtoms );

void h_generateInitialDist(double3 *pos,
                           double3 *vel,
                           double3 *acc,
                           cuDoubleComplex *d_psiUp,
                           cuDoubleComplex *d_psiDn,
                           int      numberOfAtoms,
                           double   Temp,
                           curandState_t *rngState,
                           int *atomID,
                           hbool_t *atomIsSpinUp );

__global__ void generateInitialDist(double3 *pos,
                                    double3 *vel,
                                    double3 *acc,
                                    cuDoubleComplex *psiUp,
                                    cuDoubleComplex *psiDn,
                                    int      numberOfAtoms,
                                    double   Temp,
                                    curandState_t *rngState,
                                    int *atomID,
                                    hbool_t *atomIsSpinUp );

__device__ double3 getRandomVelocity( double Temp,
                                      curandState_t *rngState );

__device__ double3 selectAtomInThermalDistribution( double Temp,
                                                    curandState_t *rngState );

__device__ double3 getGaussianPoint( double mean,
                                     double std,
                                     curandState_t *rngState );

__device__ double3 updateAccel( double3 pos );

__device__ cuDoubleComplex getAlignedSpinUp( double3 pos );

__device__ cuDoubleComplex getAlignedSpinDn( double3 pos );

__device__ double3 getMagFieldNormal( double3 pos );

__device__ double3 getMagField( double3 pos );

__device__ double3 getBdiffX( double3 pos );
__device__ double3 getBdiffY( double3 pos );
__device__ double3 getBdiffZ( double3 pos );

void initSigvrmax( double *d_sigvrmax, int numberOfCells );

#endif
