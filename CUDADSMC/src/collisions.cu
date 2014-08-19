//
//  collisions.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>

#include "collisions.cuh"

void  indexAtoms( double3 *d_pos, int *d_cellID )
{
    float *d_radius;
    
	cudaMalloc( (void **)&d_radius, numberOfAtoms*sizeof(float) );
    
    cudaMemset( d_radius, 0., numberOfAtoms*sizeof(float) );
    
    return;
}