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
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "declareInitialSystemParameters.cuh"
#include "initialSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "hdf5Helpers.cuh"
#include "setUp.cuh"
#include "moveAtoms.cuh"
#include "collisions.cuh"

char filename[] = "motionTest.h5";
char groupname[] = "/atomData";

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
	
	int sizeOfRNG = numberOfAtoms;
    
	curandStatePhilox4_32_10_t *d_rngStates;
	cudaMalloc( (void **)&d_rngStates, sizeOfRNG*sizeof(curandStatePhilox4_32_10_t) );
	
	int blockSize;
	int minGridSize;
	
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                       &blockSize,
                                       (const void *) initRNG,
                                       0,
                                       sizeOfRNG );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("initRNG:             gridSize = %i, blockSize = %i\n", gridSize, blockSize);
	initRNG<<<gridSize,blockSize>>>( d_rngStates, sizeOfRNG );
    
#pragma mark - Memory Allocation
    
    double time = 0;
    
    double3 *d_pos;
    double3 *d_vel;
    double3 *d_acc;
    
    cudaCalloc( (void **)&d_pos, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_vel, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_acc, numberOfAtoms, sizeof(double3) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) generateInitialDist,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    printf("generateInitialDist: gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    generateInitialDist<<<gridSize,blockSize>>>( d_pos,
                                                 d_vel,
                                                 d_acc,
                                                 numberOfAtoms,
                                                 Tinit,
                                                 dBdz,
                                                 d_rngStates );
    
#pragma mark - Write Initial State
    
    createHDF5File( filename,
                    groupname );
    
    cudaMemcpy( h_pos,
                d_pos,
                numberOfAtoms*sizeof(double3),
                cudaMemcpyDeviceToHost );
	char posDatasetName[] = "/atomData/positions";
    hdf5FileHandle hdf5handlePos = createHDF5Handle( numberOfAtoms,
                                                     posDatasetName );
	intialiseHDF5File( hdf5handlePos,
                       filename );
	writeHDF5File( hdf5handlePos,
                   filename,
                   h_pos );
    
    cudaMemcpy( h_vel,
                d_vel,
                numberOfAtoms*sizeof(double3),
                cudaMemcpyDeviceToHost );
    char velDatasetName[] = "/atomData/velocities";
    hdf5FileHandle hdf5handleVel = createHDF5Handle( numberOfAtoms,
                                                     velDatasetName );
	intialiseHDF5File( hdf5handleVel,
                       filename );
	writeHDF5File( hdf5handleVel,
                   filename,
                   h_vel );
    
    char timeDatasetName[] = "/atomData/simuatedTime";
    hdf5FileHandle hdf5handleTime = createHDF5HandleTime( timeDatasetName );
    intialiseHDF5File( hdf5handleTime,
                       filename );
	writeHDF5File( hdf5handleTime,
                   filename,
                   &time );
    
#pragma mark - Main Loop
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) moveAtoms,
                                        0,
                                        numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    printf("moveAtoms:           gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    for (int i=0; i<numberOfPrints; i++)
    {
        moveAtoms<<<gridSize,blockSize>>>( d_pos,
                                           d_vel,
                                           d_acc,
                                           numberOfAtoms );
        
        time += loopsPerCollision * dt;
        
        cudaMemcpy( h_pos,
                    d_pos,
                    numberOfAtoms*sizeof(double3),
                    cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel,
                    d_vel,
                    numberOfAtoms*sizeof(double3),
                    cudaMemcpyDeviceToHost );
        
        writeHDF5File( hdf5handlePos,
                       filename,
                       h_pos );
        writeHDF5File( hdf5handleVel,
                       filename,
                       h_vel );
        writeHDF5File( hdf5handleTime,
                       filename,
                       &time );
        
        if ((i/numberOfPrints*100)%5==0) {
            printf("...%f%% complete\n", (float)i/numberOfPrints*100.);
        }
    }
    
    // insert code here...
    printf("\n");
    
    free( h_pos );
    free( h_vel );
    
    cudaFree( d_pos );
    cudaFree( d_vel );
    cudaFree( d_acc );
    
    cudaDeviceReset();
    
    return 0;
}

