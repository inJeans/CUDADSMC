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

char filename[] = "outputData.h5";
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
	
	curandStatePhilox4_32_10_t *rngStates;
	cudaMalloc( (void **)&rngStates, numberOfAtoms*sizeof(curandStatePhilox4_32_10_t) );
	
	int blockSize;
	int minGridSize;
	
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) initRNG,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("initRNG:             gridSize = %i, blockSize = %i\n", gridSize, blockSize);
	initRNG<<<gridSize,blockSize>>>( rngStates, numberOfAtoms );
    
#pragma mark - Memory Allocation
    
    double3 *d_pos;
    double3 *d_vel;
    double3 *d_acc;
    
    float medianR;
    
    int2 *d_cellStartEnd;
    
    int *d_cellID;
    int *d_numberOfAtomsInCell;
    int *d_prefixScanNumberOfAtomsInCell;

    cudaCalloc( (void **)&d_pos, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_vel, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_acc, numberOfAtoms, sizeof(double3) );
    
    cudaCalloc( (void **)&d_cellStartEnd, numberOfCells+1, sizeof(int2) );
    
    cudaCalloc( (void **)&d_cellID, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_numberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_prefixScanNumberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    thrust::device_ptr<int> th_numberOfAtomsInCell = thrust::device_pointer_cast( d_numberOfAtomsInCell );
    thrust::device_ptr<int> th_prefixScanNumberOfAtomsInCell = thrust::device_pointer_cast( d_prefixScanNumberOfAtomsInCell );
    
    int *h_numberOfAtomsInCell = (int*) calloc( numberOfCells+1, sizeof(int) );
	int *h_cellID = (int*) calloc( numberOfAtoms, sizeof(int) );
    
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
                                                 rngStates );
    
    medianR = indexAtoms( d_pos,
                          d_cellID );
	sortArrays( d_pos,
                d_vel,
                d_acc,
                d_cellID );
    
    createHDF5File( filename, groupname );
    
    
    cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
	char posDatasetName[] = "/atomData/positions";
    hdf5FileHandle hdf5handlePos = createHDF5Handle( numberOfAtoms,
                                                     posDatasetName );
	intialiseHDF5File( hdf5handlePos,
                       filename );
	writeHDF5File( hdf5handlePos,
                   filename,
                   h_pos );
    
    cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
    char velDatasetName[] = "/atomData/velocities";
    hdf5FileHandle hdf5handleVel = createHDF5Handle( numberOfAtoms,
                                                     velDatasetName );
	intialiseHDF5File( hdf5handleVel,
                       filename );
	writeHDF5File( hdf5handleVel,
                   filename,
                   h_vel );
    
#pragma mark - Main Loop
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) moveAtoms,
                                        0,
                                        numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    printf("moveAtoms:           gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    for (int i=0; i<10; i++)
    {
        medianR = indexAtoms( d_pos,
                              d_cellID );
        sortArrays( d_pos,
                    d_vel,
                    d_acc,
                    d_cellID );
		
		cellStartandEndKernel<<<gridSize,blockSize>>>( d_cellID,
                                                       d_cellStartEnd,
                                                       numberOfAtoms );
        findNumberOfAtomsInCell<<<numberOfCells+1,1>>>( d_cellStartEnd,
                                                        d_numberOfAtomsInCell,
                                                        numberOfCells );
        thrust::exclusive_scan( th_numberOfAtomsInCell,
                                th_numberOfAtomsInCell + numberOfCells + 1,
                                th_prefixScanNumberOfAtomsInCell );
        
        collide<<<numberOfCells,64>>>( d_pos,
                                       d_vel,
                                       d_prefixScanNumberOfAtomsInCell,
                                       medianR,
                                       numberOfCells );
        
        moveAtoms<<<gridSize,blockSize>>>( d_pos,
                                           d_vel,
                                           d_acc,
                                           numberOfAtoms );
    
        cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
    
        writeHDF5File( hdf5handlePos,
                       filename,
                       h_pos );
        writeHDF5File( hdf5handleVel,
                       filename,
                       h_vel );
        
        printf("i = %i\n", i);
    }
    // insert code here...
    printf("\n");
    
    free( h_pos );
    free( h_vel );
    free( h_numberOfAtomsInCell );
	free( h_cellID );
    
    cudaFree( d_pos );
    cudaFree( d_vel );
    cudaFree( d_acc );
    cudaFree( d_cellStartEnd );
    cudaFree( d_cellID );
    cudaFree( d_numberOfAtomsInCell );
    cudaFree( d_prefixScanNumberOfAtomsInCell );
    cudaFree( rngStates );
    
    cudaDeviceReset();
    
    return 0;
}

