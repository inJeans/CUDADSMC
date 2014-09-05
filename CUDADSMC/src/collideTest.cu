//
//  collideTest.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 4/09/2014.
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

char filename[] = "collideTest.h5";
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
	
	if( argc == 2 )
	{
		dt = atof(argv[1]);
		printf("dt = %g\n", dt);
	}
	else if( argc > 2 )
	{
		printf("Too many arguments supplied.\n");
		return 0;
	}
	else
	{
		dt = 5.e-6;
	}
	
    loopsPerCollision = 0.001 / dt;
    
	copyConstantsToDevice<<<1,1>>>( dt );
	
    int sizeOfRNG = findRNGArrayLength( );
	
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
    
    double3 *d_pos;
    double3 *d_vel;
    double3 *d_acc;
    
    double *d_sigvrmax;
    
    double time = 0;
    double medianR;
    
    int2 *d_cellStartEnd;
    
    int *d_cellID;
    int *d_numberOfAtomsInCell;
    int *d_prefixScanNumberOfAtomsInCell;
    int *d_collisionCount;
    
    cudaCalloc( (void **)&d_pos, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_vel, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_acc, numberOfAtoms, sizeof(double3) );
    
    cudaCalloc( (void **)&d_sigvrmax, numberOfCells+1, sizeof(double) );
    initSigvrmax( d_sigvrmax, numberOfCells );
    
    cudaCalloc( (void **)&d_cellStartEnd, numberOfCells+1, sizeof(int2) );
    
    cudaCalloc( (void **)&d_cellID, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_numberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_prefixScanNumberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_collisionCount, numberOfCells+1, sizeof(int) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    int *h_numberOfAtomsInCell = (int*) calloc( numberOfCells+1, sizeof(int) );
	int *h_cellID = (int*) calloc( numberOfAtoms, sizeof(int) );
    int *h_collisionCount = (int*) calloc( numberOfCells+1, sizeof(int) );
    
    thrust::device_ptr<int> th_numberOfAtomsInCell = thrust::device_pointer_cast( d_numberOfAtomsInCell );
    thrust::device_ptr<int> th_prefixScanNumberOfAtomsInCell = thrust::device_pointer_cast( d_prefixScanNumberOfAtomsInCell );
    
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
    
    medianR = indexAtoms( d_pos,
                          d_cellID );
	sortArrays( d_pos,
                d_vel,
                d_acc,
                d_cellID );
    
#pragma mark - Write Initial State
    
    createHDF5File( filename,
                    groupname );
    
    cudaMemcpy( h_pos,
                d_pos,
                numberOfAtoms*sizeof(double3),
                cudaMemcpyDeviceToHost );
	char posDatasetName[] = "/atomData/positions";
    int3 atomDims = { numberOfAtoms, 3, 1 };
    hdf5FileHandle hdf5handlePos = createHDF5Handle( atomDims,
                                                     H5T_NATIVE_DOUBLE,
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
    hdf5FileHandle hdf5handleVel = createHDF5Handle( atomDims,
                                                     H5T_NATIVE_DOUBLE,
                                                     velDatasetName );
    intialiseHDF5File( hdf5handleVel,
                       filename );
	writeHDF5File( hdf5handleVel,
                   filename,
                   h_vel );
    
    cudaMemcpy( h_collisionCount,
                d_collisionCount,
                (numberOfCells+1)*sizeof(int),
                cudaMemcpyDeviceToHost );
	char collisionDatasetName[] = "/atomData/collisionCount";
    int3 collisionDims = { numberOfCells+1, 1, 1 };
    hdf5FileHandle hdf5handleCollision = createHDF5Handle( collisionDims,
                                                           H5T_NATIVE_INT,
                                                           collisionDatasetName );
	intialiseHDF5File( hdf5handleCollision,
                       filename );
	writeHDF5File( hdf5handleCollision,
                   filename,
                   h_collisionCount );
    
    char timeDatasetName[] = "/atomData/simuatedTime";
    int3 timeDims = { 1, 1, 1 };
    hdf5FileHandle hdf5handleTime = createHDF5Handle( timeDims,
                                                      H5T_NATIVE_DOUBLE,
                                                      timeDatasetName );
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
        medianR = indexAtoms( d_pos,
                             d_cellID );
        sortArrays( d_pos,
                   d_vel,
                   d_acc,
                   d_cellID );
		
		deviceMemset<<<numberOfCells+1,1>>>( d_cellStartEnd,
                                            make_int2( -1, -1 ),
                                            numberOfCells + 1 );
		cellStartandEndKernel<<<gridSize,blockSize>>>( d_cellID,
                                                      d_cellStartEnd,
                                                      numberOfAtoms );
        findNumberOfAtomsInCell<<<numberOfCells+1,1>>>( d_cellStartEnd,
                                                       d_numberOfAtomsInCell,
                                                       numberOfCells );
        thrust::exclusive_scan( th_numberOfAtomsInCell,
                               th_numberOfAtomsInCell + numberOfCells + 1,
                               th_prefixScanNumberOfAtomsInCell );
        
        collide<<<numberOfCells,64>>>( d_vel,
                                       d_sigvrmax,
                                       d_prefixScanNumberOfAtomsInCell,
                                       d_collisionCount,
                                       medianR,
                                       numberOfCells,
                                       d_rngStates );
        
        moveAtoms<<<gridSize,blockSize>>>( d_pos,
                                          d_vel,
                                          d_acc,
                                          numberOfAtoms );
        
        time += loopsPerCollision * dt;
        
        cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_collisionCount, d_collisionCount, (numberOfCells+1)*sizeof(int), cudaMemcpyDeviceToHost );
        
        writeHDF5File( hdf5handlePos,
                      filename,
                      h_pos );
        writeHDF5File( hdf5handleVel,
                      filename,
                      h_vel );
        writeHDF5File( hdf5handleCollision,
                       filename,
                       h_collisionCount );
        writeHDF5File( hdf5handleTime,
                       filename,
                       &time );
        
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
    cudaFree( d_sigvrmax );
    cudaFree( d_cellStartEnd );
    cudaFree( d_cellID );
    cudaFree( d_numberOfAtomsInCell );
    cudaFree( d_prefixScanNumberOfAtomsInCell );
    cudaFree( d_rngStates );
    
    cudaDeviceReset();
    
    return 0;
}