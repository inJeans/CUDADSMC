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
#pragma mark - Read commandline arguments
    int    numberOfAtoms;
    int    numberOfCells;
    int3   cellsPerDimension;
    double alpha;
    
    if ( argc < 3 )
    {
        int Nc = 50;
        cellsPerDimension = make_int3( Nc, Nc, Nc );
        numberOfCells = Nc*Nc*Nc;
        
        numberOfAtoms = 1e4;
        alpha = 1e6 / numberOfAtoms;
    }
    else if ( argc == 3 )
    {
        int Nc = atoi(argv[1]);
        cellsPerDimension = make_int3( Nc, Nc, Nc );
        numberOfCells = Nc*Nc*Nc;
        
        numberOfAtoms = atoi(argv[2]);
        alpha = 1e6 / numberOfAtoms;
    }
    else if( argc > 3 )
    {
        printf("Too many arguments supplied.\n");
        return 0;
    }
    
#pragma mark - Set up CUDA device
	// Flush device (useful for profiling)
    cudaDeviceReset();
	
	int maxDevice = 0;
	maxDevice = setMaxCUDADevice( );
	
	int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, maxDevice);
    
#pragma mark - Memory Allocation
    
    int sizeOfRNG = findRNGArrayLength( numberOfCells, numberOfAtoms );
    printf("%i\n", sizeOfRNG);
	curandState_t *d_rngStates;
	cudaMalloc( (void **)&d_rngStates, sizeOfRNG*sizeof(curandState_t) );
    
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
    int *d_atomID;
    
    hbool_t *d_isPerturb;

    cudaCalloc( (void **)&d_pos, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_vel, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_acc, numberOfAtoms, sizeof(double3) );
    
    cudaCalloc( (void **)&d_sigvrmax, numberOfCells+1, sizeof(double) );
    
    cudaCalloc( (void **)&d_cellStartEnd, numberOfCells+1, sizeof(int2) );
    
    cudaCalloc( (void **)&d_cellID, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_numberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_prefixScanNumberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_collisionCount, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_atomID, numberOfAtoms, sizeof(int) );
    
    cudaCalloc( (void **)&d_isPerturb, numberOfAtoms, sizeof(hbool_t) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    int *h_numberOfAtomsInCell = (int*) calloc( numberOfCells+1, sizeof(int) );
    int *h_collisionCount = (int*) calloc( numberOfCells+1, sizeof(int) );
	int *h_cellID = (int*) calloc( numberOfAtoms, sizeof(int) );
    int *h_atomID = (int*) calloc( numberOfAtoms, sizeof(int) );
    
    hbool_t *h_isPerturb = (hbool_t*) calloc( numberOfAtoms, sizeof(hbool_t) );
    
    thrust::device_ptr<int> th_numberOfAtomsInCell = thrust::device_pointer_cast( d_numberOfAtomsInCell );
    thrust::device_ptr<int> th_prefixScanNumberOfAtomsInCell = thrust::device_pointer_cast( d_prefixScanNumberOfAtomsInCell );
    
#pragma mark - Set up atom system
	
    dt = 1.e-6;
    loopsPerCollision = 0.025 / dt;
    
	copyConstantsToDevice<<<1,1>>>( dt );
	
    h_initRNG( d_rngStates, numberOfAtoms );
    
    h_generateInitialDist( d_pos,
                           d_vel,
                           d_acc,
                           numberOfAtoms,
                           Tinit,
                           d_rngStates,
                           d_isPerturb,
                           d_atomID );
    
    initSigvrmax( d_sigvrmax, numberOfCells );
    
    medianR = indexAtoms( d_pos,
                          d_cellID,
                          cellsPerDimension,
                          numberOfAtoms );
    sortArrays( d_pos,
                d_vel,
                d_acc,
                d_cellID,
                d_isPerturb,
                d_atomID,
                numberOfAtoms );
    
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
    
    cudaMemcpy( h_isPerturb,
                d_isPerturb,
                numberOfAtoms*sizeof(hbool_t),
                cudaMemcpyDeviceToHost );
    char isPerturbDatasetName[] = "/atomData/isPerturb";
    int3 perturbDims = { numberOfAtoms, 1, 1 };
    hdf5FileHandle hdf5handlePerturb = createHDF5Handle( perturbDims,
                                                         H5T_NATIVE_HBOOL,
                                                         isPerturbDatasetName );
    intialiseHDF5File( hdf5handlePerturb,
                       filename );
    writeHDF5File( hdf5handlePerturb,
                   filename,
                   h_isPerturb );
    
    cudaMemcpy( h_cellID,
               d_cellID,
               numberOfAtoms*sizeof(int),
               cudaMemcpyDeviceToHost );
    char cellIDDatasetName[] = "/atomData/cellID";
    hdf5FileHandle hdf5handleCellID = createHDF5Handle( perturbDims,
                                                        H5T_NATIVE_INT,
                                                        cellIDDatasetName );
    intialiseHDF5File( hdf5handleCellID,
                       filename );
    writeHDF5File( hdf5handleCellID,
                   filename,
                   h_cellID );
    
    cudaMemcpy( h_atomID,
               d_atomID,
               numberOfAtoms*sizeof(int),
               cudaMemcpyDeviceToHost );
    char atomIDDatasetName[] = "/atomData/atomID";
    hdf5FileHandle hdf5handleAtomID = createHDF5Handle( perturbDims,
                                                       H5T_NATIVE_INT,
                                                       atomIDDatasetName );
    intialiseHDF5File( hdf5handleAtomID,
                      filename );
    writeHDF5File( hdf5handleAtomID,
                  filename,
                  h_atomID );
    
#pragma mark - Main Loop
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) moveAtoms,
                                        0,
                                        sizeOfRNG );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    cudaDeviceGetAttribute( &numSMs,
                            cudaDevAttrMultiProcessorCount,
                            device);
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    printf("blocksize = %i, gridsize = %i\n", blockSize, gridSize);
    
    for (int i=0; i<numberOfPrints; i++)
    {
#pragma mark Collide Atoms
        
        medianR = indexAtoms( d_pos,
                              d_cellID,
                              cellsPerDimension,
                              numberOfAtoms );
        
        sortArrays( d_pos,
                    d_vel,
                    d_acc,
                    d_cellID,
                    d_isPerturb,
                    d_atomID,
                    numberOfAtoms );
        
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
                                      d_prefixScanNumberOfAtomsInCell,
                                      d_collisionCount,
                                      medianR,
                                      alpha,
                                      cellsPerDimension,
                                      numberOfCells,
                                      d_rngStates,
                                      d_cellID,
                                      d_atomID );
        
#pragma mark Evolve System
        
        for (int j=0; j<loopsPerCollision; j++) {
            
            moveAtoms<<<gridSize,blockSize>>>( d_pos,
                                               d_vel,
                                               d_acc,
                                               numberOfAtoms );
        }
        
        printf( "Number of atoms = %i, ", numberOfAtoms);
        
        time += loopsPerCollision * dt;
    
        cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_numberOfAtomsInCell, d_numberOfAtomsInCell, (numberOfCells+1)*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_collisionCount, d_collisionCount, (numberOfCells+1)*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_isPerturb, d_isPerturb, numberOfAtoms*sizeof(hbool_t), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_cellID, d_cellID, numberOfAtoms*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_atomID, d_atomID, numberOfAtoms*sizeof(int), cudaMemcpyDeviceToHost );
        
        writeHDF5File( hdf5handlePos,
                       filename,
                       h_pos );
        writeHDF5File( hdf5handleVel,
                       filename,
                       h_vel );
        writeHDF5File( hdf5handleCollision,
                       filename,
                       h_collisionCount );
        writeHDF5File( hdf5handlenAtom,
                       filename,
                       h_numberOfAtomsInCell );
        writeHDF5File( hdf5handleTime,
                       filename,
                       &time );
        writeHDF5File( hdf5handleNumber,
                       filename,
                       &numberOfAtoms );
        writeHDF5File( hdf5handlePerturb,
                       filename,
                       h_isPerturb );
        writeHDF5File( hdf5handleCellID,
                       filename,
                       h_cellID );
        writeHDF5File( hdf5handleAtomID,
                       filename,
                       h_atomID );
        
        printf("i = %i\n", i);
    }
    
    // insert code here...
    printf("\n");
    
    free( h_pos );
    free( h_vel );
    free( h_numberOfAtomsInCell );
	free( h_cellID );
    free( h_isPerturb );
    
    cudaFree( d_pos );
    cudaFree( d_vel );
    cudaFree( d_acc );
    cudaFree( d_sigvrmax );
    cudaFree( d_cellStartEnd );
    cudaFree( d_cellID );
    cudaFree( d_numberOfAtomsInCell );
    cudaFree( d_prefixScanNumberOfAtomsInCell );
    cudaFree( d_collisionCount );
    cudaFree( d_rngStates );
    cudaFree( d_isPerturb );
    
    cudaDeviceReset();
    
    return 0;
}

