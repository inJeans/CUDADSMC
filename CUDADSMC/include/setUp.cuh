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

__global__ void initRNG( curandStatePhilox4_32_10_t *rngState,
                         int numberOfAtoms );

__global__ void generateInitialDist(double4 *pos,
                                    double4 *vel,
                                    int      numberOfAtoms,
									double   Temp,
									double   dBdz,
									curandStatePhilox4_32_10_t *rngState);

__device__ double4 getRandomVelocity( double Temp,
                                      curandStatePhilox4_32_10_t *rngState );

__device__ double4 getRandomPointOnUnitSphere( curandStatePhilox4_32_10_t *rngState );

__device__ double4 selectAtomInDistribution( double dBdz,
                                             double Temp,
                                             curandStatePhilox4_32_10_t *rngState );

__device__ double4 getGaussianPoint( double mean,
                                     double std,
                                     curandStatePhilox4_32_10_t *rngState );

__device__ bool pointIsInDistribution( double4 point,
                                       double dBdz,
                                       double Temp,
                                       curandStatePhilox4_32_10_t *rngState );

#endif
