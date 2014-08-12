//
//  statTests.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 9/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <thrust/sort.h>

#include "statTests.cuh"

void shapiroWilk( double *data, int N )
{
	
	if ( N < 3) {
		fprintf( stderr, "Sample vector must have at least 3 observations.\n" );
        exit( EXIT_FAILURE ); /* indicate failure.*/
	}
	else if ( N > 5000 ) {
		printf(" BE CAREFUL - Shapiro-Wilk statistic might be inaccurate due to large sample size (> 5000).\n" );
	}
	
	thrust::sort( data, data + N );
	
	double *d_m;
    cudaMalloc( (void **)&d_m, N*sizeof(double) );
	
	int blockSize;
	int minGridSize;
	
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
									    &blockSize,
									    calcm,
									    0,
									    N );
	
	gridSize = ( N + blockSize - 1 ) / blockSize;
	
	calcm<<<gridSize,blockSize>>>( d_m, N );
	
	cublasStatus_t cublasDdot (cublasHandle_t handle, int n, d_m, 1, d_m, 1, double *result);
	
	return;
}

__global__ void calcm( double *m, int N )
{
	for (int n = blockIdx.x * blockDim.x + threadIdx.x;
		 n < N;
		 n += blockDim.x * gridDim.x)
	{
		double p = ( (n+1.) - 3./8. ) / ( N + 0.25 );
		
		m[n] = normcdfinv( p );
	}
}