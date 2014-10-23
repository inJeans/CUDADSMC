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

int findRNGArrayLength( int numberOfCells );

void h_initRNG( curandStatePhilox4_32_10_t *rngState,
                int numberOfAtoms );

__global__ void initRNG( curandStatePhilox4_32_10_t *rngState,
                         int numberOfAtoms );

void h_generateInitialDist( double3 *pos,
                            double3 *vel,
                            double3 *acc,
                            int      numberOfAtoms,
                            double   Temp,
                            curandStatePhilox4_32_10_t *rngState );

__global__ void generateInitialDist(double3 *pos,
                                    double3 *vel,
                                    double3 *acc,
                                    int      numberOfAtoms,
									double   Temp,
									curandStatePhilox4_32_10_t *rngState);

__device__ double3 getRandomVelocity( double Temp,
                                      curandStatePhilox4_32_10_t *rngState );

__device__ double3 selectAtomInThermalDistribution( double Temp,
                                                    curandStatePhilox4_32_10_t *rngState );

__device__ double3 getGaussianPoint( double mean,
                                     double std,
                                     curandStatePhilox4_32_10_t *rngState );

__device__ double3 updateAccel( double3 pos );

void initSigvrmax( double *d_sigvrmax, int numberOfCells );

#endif
