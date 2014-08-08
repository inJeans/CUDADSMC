//
//  initTest.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 7/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#include "initialSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "setUp.cuh"

int main(int argc, const char * argv[])
{
#pragma mark - Set up CUDA device
	// Flush device (useful for profiling)
    cudaDeviceReset();
	
	int maxDevice = 0;
	maxDevice = setMaxCUDADevice( );
	
	int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, maxDevice);
	
#pragma mark - Set up atom system
	
	curandStatePhilox4_32_10_t *rngStates;
	cudaMalloc( (void **)&rngStates, numberOfAtoms*sizeof(curandStatePhilox4_32_10_t) );
	
	int blockSize;
	int minGridSize;
	
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
									   &blockSize,
									   initRNG,
									   0,
									   numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	
	initRNG<<<gridSize,blockSize>>>( rngStates, numberOfAtoms );
    
    double4 *d_pos;
    double4 *d_vel;
    
	cudaMalloc( (void **)&d_pos, numberOfAtoms*sizeof(double4) );
    cudaMalloc( (void **)&d_vel, numberOfAtoms*sizeof(double4) );
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
									   &blockSize,
									   generateInitialDist,
									   0,
									   numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    
    generateInitialDist<<<gridSize,blockSize>>>( d_pos,
												d_vel,
												numberOfAtoms,
												Tinit,
												dBdz,
												rngStates );
    
    printf("gridSize = %i, blockSize = %i\n", gridSize, blockSize);
	
	double4 *h_pos = (*double4) calloc( numberOfAtoms, sizeof(double4) );
	double4 *h_vel = (*double4) calloc( numberOfAtoms, sizeof(double4) );
	
	cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double4), cudaMemcpyDeviceToHost );
	cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double4), cudaMemcpyDeviceToHost );
	
	
    // insert code here...
    printf("Hello, World!\n");
    return 0;
}
