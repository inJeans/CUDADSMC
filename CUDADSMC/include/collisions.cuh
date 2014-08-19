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


#endif
