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
//#include "declareDeviceSystemParameters.cuh"
#include "deviceSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "hdf5Helpers.cuh"
#include "vectorMath.cuh"
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
    
#pragma mark - Memory Allocation
    
    int sizeOfRNG = findRNGArrayLength( );
	
	curandStatePhilox4_32_10_t *d_rngStates;
	cudaMalloc( (void **)&d_rngStates, sizeOfRNG*sizeof(curandStatePhilox4_32_10_t) );
    
    double3 *d_pos;
    double3 *d_vel;
    double3 *d_acc;
    
    double *d_sigvrmax;
    
    double time = 0.;
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
    
    cudaCalloc( (void **)&d_cellStartEnd, numberOfCells+1, sizeof(int2) );
    
    cudaCalloc( (void **)&d_cellID, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_numberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_prefixScanNumberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_collisionCount, numberOfCells+1, sizeof(int) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    int *h_numberOfAtomsInCell = (int*) calloc( numberOfCells+1, sizeof(int) );
    int *h_collisionCount = (int*) calloc( numberOfCells+1, sizeof(int) );
	int *h_cellID = (int*) calloc( numberOfAtoms, sizeof(int) );
    
    thrust::device_ptr<int> th_numberOfAtomsInCell = thrust::device_pointer_cast( d_numberOfAtomsInCell );
    thrust::device_ptr<int> th_prefixScanNumberOfAtomsInCell = thrust::device_pointer_cast( d_prefixScanNumberOfAtomsInCell );
    
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
		dt = 1.0e-6;
	}
	
    loopsPerCollision = 0.005 / dt;
    
	copyConstantsToDevice<<<1,1>>>( dt );
	
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
    
    initSigvrmax( d_sigvrmax, numberOfCells );
    
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
    
    cudaMemcpy( h_numberOfAtomsInCell,
                d_numberOfAtomsInCell,
                (numberOfCells+1)*sizeof(int),
                cudaMemcpyDeviceToHost );
	char nAtomDatasetName[] = "/atomData/atomCount";
    int3 nAtomDims = { numberOfCells+1, 1, 1 };
    hdf5FileHandle hdf5handlenAtom = createHDF5Handle( nAtomDims,
                                                       H5T_NATIVE_INT,
                                                       nAtomDatasetName );
	intialiseHDF5File( hdf5handlenAtom,
                       filename );
	writeHDF5File( hdf5handlenAtom,
                   filename,
                   h_numberOfAtomsInCell );
    
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
    
    char numberDatasetName[] = "/atomData/atomNumber";
    hdf5FileHandle hdf5handleNumber = createHDF5Handle( timeDims,
                                                      H5T_NATIVE_INT,
                                                      numberDatasetName );
    intialiseHDF5File( hdf5handleNumber,
                       filename );
    writeHDF5File( hdf5handleNumber,
                   filename,
                   &numberOfAtoms );
    
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
#pragma mark Collide Atoms
        
        medianR = indexAtoms( d_pos,
                              d_cellID );
        
        sortArrays( d_pos,
                    d_vel,
                    d_acc,
                    d_psiU,
                    d_psiD,
                    d_oldPops2,
                    d_isSpinUp,
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
        
        collide<<<numberOfCells,1>>>( d_vel,
                                       d_sigvrmax,
                                       d_isSpinUp,
                                       d_prefixScanNumberOfAtomsInCell,
                                       d_collisionCount,
                                       medianR,
                                       numberOfCells,
                                       d_rngStates,
                                       d_cellID );
        
#pragma mark Evolve System
        
        for (int j=0; j<loopsPerCollision; j++) {
            
//            unitaryEvolution<<<gridSize,blockSize>>>( d_psiU,
//                                                      d_psiD,
//                                                      d_oldPops2,
//                                                      d_pos,
//                                                      d_vel,
//                                                      numberOfAtoms );
            
            moveAtoms<<<gridSize,blockSize>>>( d_pos,
                                               d_vel,
                                               d_acc,
                                               numberOfAtoms,
                                               d_isSpinUp,
                                               medianR );
            
//            projectSpins<<<gridSize,blockSize>>>( d_psiU,
//                                                  d_psiD,
//                                                  d_oldPops2,
//                                                  d_pos,
//                                                  d_vel,
//                                                  d_isSpinUp,
//                                                  d_rngStates,
//                                                  numberOfAtoms,
//                                                  d_flippedPos,
//                                                  d_flippedVel );
            
//            exponentialDecay<<<gridSize,blockSize>>>( d_psiU,
//                                                      d_psiD,
//                                                      d_pos,
//                                                      d_isSpinUp,
//                                                      numberOfAtoms );
//
//            normaliseWavefunction<<<gridSize,blockSize>>>( d_psiU,
//                                                           d_psiD,
//                                                           numberOfAtoms );
        }
        
//#pragma mark Evaoprate Atoms
//        
//        evaporateAtoms( d_pos,
//                        d_vel,
//                        d_acc,
//                        d_psiU,
//                        d_psiD,
//                        d_oldPops2,
//                        d_isSpinUp,
//                        d_cellID,
//                        medianR,
//                        &numberOfAtoms );
        
        printf( "Number of atoms = %i, ", numberOfAtoms);
        
        time += loopsPerCollision * dt;
    
        cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_psiU, d_psiU, numberOfAtoms*sizeof(zomplex), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_psiD, d_psiD, numberOfAtoms*sizeof(zomplex), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_isSpinUp, d_isSpinUp, numberOfAtoms*sizeof(hbool_t), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_collisionCount, d_collisionCount, (numberOfCells+1)*sizeof(int), cudaMemcpyDeviceToHost );
    
        writeHDF5File( hdf5handlePos,
                       filename,
                       h_pos );
        writeHDF5File( hdf5handleVel,
                       filename,
                       h_vel );
        writeHDF5File( hdf5handlePsiU,
                       filename,
                       h_psiU );
        writeHDF5File( hdf5handlePsiD,
                       filename,
                       h_psiD );
        writeHDF5File( hdf5handleIsSpinUp,
                       filename,
                       h_isSpinUp );
        writeHDF5File( hdf5handleCollision,
                       filename,
                       h_collisionCount );
        writeHDF5File( hdf5handleTime,
                       filename,
                       &time );
        writeHDF5File( hdf5handleNumber,
                       filename,
                       &numberOfAtoms );
        
        cudaMemcpy( h_flippedPos, d_flippedPos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_flippedVel, d_flippedVel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        
        writeHDF5File( hdf5handlefPos,
                       filename,
                       h_flippedPos );
        writeHDF5File( hdf5handlefVel,
                       filename,
                       h_flippedVel );
        
        printf("i = %i\n", i);
    }
    
    // insert code here...
    printf("\n");
    
    free( h_pos );
    free( h_vel );
    free( h_psiU );
    free( h_psiD );
    free( h_numberOfAtomsInCell );
	free( h_cellID );
    free( h_isSpinUp );
    
    free( h_flippedPos );
    free( h_flippedVel );
    
    cudaFree( d_pos );
    cudaFree( d_vel );
    cudaFree( d_acc );
    cudaFree( d_psiU );
    cudaFree( d_psiD );
    cudaFree( d_oldPops2 );
    cudaFree( d_sigvrmax );
    cudaFree( d_cellStartEnd );
    cudaFree( d_cellID );
    cudaFree( d_numberOfAtomsInCell );
    cudaFree( d_prefixScanNumberOfAtomsInCell );
    cudaFree( d_rngStates );
    cudaFree( d_isSpinUp );
    
    cudaFree( d_flippedPos );
    cudaFree( d_flippedVel );
    
    cudaDeviceReset();
    
    return 0;
}

