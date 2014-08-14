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

#include "hdf5.h"
#include "hdf5_hl.h"

#include "initialSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "setUp.cuh"
#include "moveAtoms.cuh"

#define RANK 2

hsize_t dims[RANK]= {numberOfAtoms, 4};

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
    
    double4 *d_pos;
    double4 *d_vel;
    double4 *d_acc;
    
	cudaMalloc( (void **)&d_pos, numberOfAtoms*sizeof(double4) );
    cudaMalloc( (void **)&d_vel, numberOfAtoms*sizeof(double4) );
    cudaMalloc( (void **)&d_acc, numberOfAtoms*sizeof(double4) );
    
    cudaMemset( d_pos, 0., numberOfAtoms*sizeof(double4) );
    cudaMemset( d_vel, 0., numberOfAtoms*sizeof(double4) );
    cudaMemset( d_acc, 0., numberOfAtoms*sizeof(double4) );
    
    double4 *h_pos = (double4*) calloc( numberOfAtoms, sizeof(double4) );
    
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
	
    cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double4), cudaMemcpyDeviceToHost );
    
    /* create a HDF5 file */
    hid_t file_id = H5Fcreate ( "beforeMove.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    /* create and write an integer type dataset named "dset" */
    H5LTmake_dataset( file_id, "/dset", RANK, dims, H5T_NATIVE_DOUBLE, h_pos );
    /* close file */
    H5Fclose ( file_id );

#pragma mark - Moving atoms
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                       &blockSize,
                                       generateInitialDist,
                                       0,
                                       numberOfAtoms );
	
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;

    
    moveAtoms<<<gridSize,blockSize>>>( d_pos,
                                       d_vel,
                                       d_acc,
                                       numberOfAtoms );
    
    
    printf("gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double4), cudaMemcpyDeviceToHost );
    
    /* create a HDF5 file */
    hid_t file_id2 = H5Fcreate ( "afterMove.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    /* create and write an integer type dataset named "dset" */
    H5LTmake_dataset( file_id2, "/dset", RANK, dims, H5T_NATIVE_DOUBLE, h_pos );
    /* close file */
    H5Fclose ( file_id2 );
    
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

