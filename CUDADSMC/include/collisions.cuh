//
//  collisions.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_collisions_cuh
#define CUDADSMC_collisions_cuh

void  indexAtoms( double3 *d_pos, int *d_cellID );
__global__ void calculateRadius( double3 *pos, float *radius, int numberOfAtoms );
float findMedian( float *v, int N );
__global__ void getMedian( float *v, float *median, int numberOfAtoms);
__global__ void findAtomIndex( double3 *pos, int *cellID, float medianR, int numberOfAtoms );
__device__ int3 getCellIndices( double3 pos, float medianR );
__device__ int getCellID( int3 index );
void sortArrays( double3 *d_pos, double3 *d_vel, double3 *d_acc, int *d_cellID );

#endif
