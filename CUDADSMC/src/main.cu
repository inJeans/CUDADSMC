//
//  main.c
//  CUDADSMC
//
//  Created by Christopher Watkins on 31/07/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#include "initialSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "hdf5Helpers.cuh"
#include "setUp.cuh"
#include "moveAtoms.cuh"

char filename[] = "outputData.h5";

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
	
	printf("gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    double3 *d_pos;
    double3 *d_vel;
    double3 *d_acc;
    
	cudaMalloc( (void **)&d_pos, numberOfAtoms*sizeof(double3) );
    cudaMalloc( (void **)&d_vel, numberOfAtoms*sizeof(double3) );
    cudaMalloc( (void **)&d_acc, numberOfAtoms*sizeof(double3) );
    
    cudaMemset( d_pos, 0., numberOfAtoms*sizeof(double3) );
    cudaMemset( d_vel, 0., numberOfAtoms*sizeof(double3) );
    cudaMemset( d_acc, 0., numberOfAtoms*sizeof(double3) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        generateInitialDist,
                                        0,
                                        numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    
    generateInitialDist<<<gridSize,blockSize>>>( d_pos,
                                                 d_vel,
                                                 d_acc,
                                                 numberOfAtoms,
                                                 Tinit,
                                                 dBdz,
                                                 rngStates );
    
    printf("gridSize = %i, blockSize = %i\n", gridSize, blockSize);
	
    createHDF5File( filename );
    
    cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
	hdf5FileHandle hdf5handlePos = createHDF5Handle( numberOfAtoms, "/positions" );
	intialiseHDF5File( hdf5handlePos, filename );
	writeHDF5File( hdf5handlePos, filename, h_pos );
    
    cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
    hdf5FileHandle hdf5handleVel = createHDF5Handle( numberOfAtoms, "/velocities" );
	intialiseHDF5File( hdf5handleVel, filename );
	writeHDF5File( hdf5handleVel, filename, h_vel );

#pragma mark - Moving atoms
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                       &blockSize,
                                       generateInitialDist,
                                       0,
                                       numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    printf("gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    for (int i=0; i<5; i++)
    {
        moveAtoms<<<gridSize,blockSize>>>(d_pos,
                                          d_vel,
                                          d_acc,
                                          numberOfAtoms );
    
        cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
    
        writeHDF5File( hdf5handlePos, filename, h_pos );
        writeHDF5File( hdf5handleVel, filename, h_vel );
        
        printf("i = %i\n", i);
    }
    // insert code here...
    printf("\n");
    
    free( h_pos );
    
    cudaFree( d_pos );
    cudaFree( d_vel );
    cudaFree( d_acc );
    cudaFree( rngStates );
    
    cudaDeviceReset();
    
    return 0;
}

