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
#include <curand_kernel.h>
#include <hdf5.h>

int findRNGArrayLength( void );

__global__ void initRNG( curandStatePhilox4_32_10_t *rngState,
                         int numberOfAtoms );

__global__ void generateInitialDist(double3 *pos,
                                    double3 *vel,
                                    double3 *acc,
                                    hbool_t *isSpinUp,
                                    int      numberOfAtoms,
									double   Temp,
									double   dBdz,
									curandStatePhilox4_32_10_t *rngState);

__device__ double3 getRandomVelocity( double Temp,
                                      curandStatePhilox4_32_10_t *rngState );

__device__ double3 getRandomPointOnUnitSphere( curandStatePhilox4_32_10_t *rngState );

__device__ double3 selectAtomInDistribution( double dBdz,
                                             double Temp,
                                             curandStatePhilox4_32_10_t *rngState );

__device__ double3 getGaussianPoint( double mean,
                                     double std,
                                     curandStatePhilox4_32_10_t *rngState );

__device__ bool pointIsInDistribution( double3 point,
                                       double dBdz,
                                       double Temp,
                                       curandStatePhilox4_32_10_t *rngState );

__device__ double3 updateAccel( double3 pos );

void initSigvrmax( double *d_sigvrmax, int numberOfCells );

void setInitialWavefunction( zomplex *d_psiU, zomplex *d_psiD, double2 *d_oldPops2, hbool_t *d_isSpinUp, double3 *d_pos, int numberOfAtoms );

__global__ void deviceSetInitialWavefunction( zomplex *psiU, zomplex *psiD, double2 *oldPops2, hbool_t *isSpinUp, double3 *pos, int numberOfAtoms );

__device__ double3 magneticField( double3 pos );

__device__ double3 magneticFieldNormal( double3 pos );

__device__ double2 getEigenStatePops( zomplex psiD, zomplex psiU, double3 Bn );

#endif
