//
//  collisions.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "declareInitialSystemParameters.cuh"
#include "collisions.cuh"
#include "vectorMath.cuh"
#include "cudaHelpers.cuh"

void  indexAtoms( double3 *d_pos, int *d_cellID )
{
    float *d_radius;
    cudaCalloc( (void **)&d_radius, numberOfAtoms, sizeof(float) );
    
    int blockSize;
	int minGridSize;
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        calculateRadius,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("calculateRadius:     gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
	calculateRadius<<<gridSize,blockSize>>>( d_pos,
                                             d_radius,
                                             numberOfAtoms );
    
    float medianR = findMedian( d_radius,
                               numberOfAtoms );
    
    printf("The median radius is %f\n", medianR );
    
    cudaFree( d_radius );
    
    return;
}

__global__ void calculateRadius( double3 *pos, float *radius, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        radius[atom] = lengthf(pos[atom]);
    }
    
    return;
}

float findMedian( float *v, int N )
{
    thrust::device_ptr<float> ptr = thrust::device_pointer_cast( v );
    
    thrust::sort( ptr,
                  ptr + N );
    
    float *d_median;
    cudaCalloc( (void **)&d_median, 1, sizeof(float) );
    
    getMedian<<<1,1>>>( v, d_median, N );
    
    float h_median;
    
    cudaMemcpy( (void *)&h_median, d_median, 1*sizeof(float), cudaMemcpyDeviceToHost );
    
    cudaFree( d_median );
    
    return h_median;
}

__global__ void getMedian( float *v, float *median, int N)
{
    if (N % 2 == 0) {
        median[0] = 0.5*(v[N/2-1] + v[N/2]);
    }
    else {
        median[0] = v[(N-1)/2];
    }
    
    return;
}