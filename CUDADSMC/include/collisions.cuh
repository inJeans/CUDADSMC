//
//  collisions.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_collisions_cuh
#define CUDADSMC_collisions_cuh

#include "vectorMath.cuh"
#include "hdf5.h"

double indexAtoms( double3 *d_pos, int *d_cellID, int3 cellsPerDimension );
void h_calculateRadius( double3 *pos, double *radius, int numberOfAtoms );
__global__ void calculateRadius( double3 *pos, double *radius, int numberOfAtoms );
double findMedian( double *v, int N );
__global__ void getMedian( double *v, double *median, int numberOfAtoms);
void h_findAtomIndex( double3 *pos, int *cellID, double medianR, int numberOfAtoms, int3 cellsPerDimension );
__global__ void findAtomIndex( double3 *pos, int *cellID, double medianR, int numberOfAtoms, int3 cellsPerDimension );
__device__ int3 getCellIndices( double3 pos, double3 gridMin, double3 cellLength );
__device__ double3 getCellLength( double medianR, int3 cellsPerDimension );
__device__ int getCellID( int3 index, int3 cellsPerDimension );
__device__ double3 getGridMin( double medianR );
__global__ void cellStartandEndKernel( int *cellID, int2 *cellStartEnd, int numberOfAtoms );
__device__ void serialCellStartandEndKernel( int *cellID, int2 *cellStartEnd, int numberOfAtoms );
__global__ void findNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells );
__device__ void serialFindNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells );
void sortArrays( double3 *d_pos,
                double3 *d_vel,
                double3 *d_acc,
                int *d_cellID );
__global__ void collide( double3 *vel,
                         double  *sigvrmax,
                         int     *prefixScanNumberOfAtomsInCell,
                         int     *collisionCount,
                         double   medianR,
                         double   alpha,
                         int3     cellsPerDimension,
                         int      numberOfCells,
                         curandStatePhilox4_32_10_t *rngState,
                         int *cellID );
__device__ int2 chooseCollidingAtoms( int numberOfAtomsInCell, int *prefixScanNumberOfAtomsInCell, int3 cellsPerDimension, curandStatePhilox4_32_10_t *rngState, int cell );
__device__ int3 extractCellIndices( int cell, int3 cellsPerDimension );
__device__ double calculateRelativeVelocity( double3 *vel, int2 collidingAtoms );
__device__ double3 getRandomPointOnSphere( curandStatePhilox4_32_10_t *rngState );

__global__ void shmemTest( double3 *vel,
                          double3 *vout,
                          int     *prefixScanNumberOfAtomsInCell,
                          curandStatePhilox4_32_10_t *rngState );

#endif
