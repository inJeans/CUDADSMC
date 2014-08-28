//
//  collisions.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_collisions_cuh
#define CUDADSMC_collisions_cuh

double  indexAtoms( double3 *d_pos, int *d_cellID );
__global__ void calculateRadius( double3 *pos, double *radius, int numberOfAtoms );
double findMedian( double *v, int N );
__global__ void getMedian( double *v, double *median, int numberOfAtoms);
__global__ void findAtomIndex( double3 *pos, int *cellID, double medianR, int numberOfAtoms );
__device__ int3 getCellIndices( double3 pos, double3 gridMin, double3 cellLength );
__device__ int getCellID( int3 index, int3 cellsPerDimensions );
__device__ double3 getGridMin( double medianR );
__global__ void cellStartandEndKernel( int *cellID, int2 *cellStartEnd, int numberOfAtoms );
__device__ void serialCellStartandEndKernel( int *cellID, int2 *cellStartEnd, int numberOfAtoms );
__global__ void findNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells );
__device__ void serialFindNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells );
void sortArrays( double3 *d_pos, double3 *d_vel, double3 *d_acc, int *d_cellID );
__global__ void collide( double3 *pos,
                        double3 *vel,
                        double  *sigvrmax,
                        int     *prefixScanNumberOfAtomsInCell,
                        double   medianR,
                        int      numberOfCells,
                        curandStatePhilox4_32_10_t *rngState );
__device__ int2 chooseCollidingAtoms( int numberOfAtomsInCell, curandStatePhilox4_32_10_t *rngState );

#endif
