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
#include "math.h"
#include <cuComplex.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/remove.h>

#include "declareInitialSystemParameters.cuh"
#include "initialSystemParameters.cuh"
//#include "declareDeviceSystemParameters.cuh"
#include "deviceSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "hdf5Helpers.cuh"
#include "vectorMath.cuh"
#include "setUp.cuh"
#include "moveAtoms.cuh"
#include "wavefunction.cuh"
#include "collisions.cuh"
#include "evaporation.cuh"

char filename[] = "outputData.h5";
char groupname[] = "/atomData";

int main(int argc, const char * argv[])
{
#pragma mark - Read commandline arguments
    int    initialNumberOfAtoms;
    int    numberOfAtoms;
    int    numberOfCells;
    int3   cellsPerDimension;
    double alpha;
    
    if ( argc < 3 )
    {
        int Nc = 50;
        cellsPerDimension = make_int3( Nc, Nc, Nc );
        numberOfCells = Nc*Nc*Nc;
        
        initialNumberOfAtoms = 1e4;
        numberOfAtoms = initialNumberOfAtoms;
        alpha = h_N / numberOfAtoms;
    }
    else if ( argc == 3 )
    {
        int Nc = atoi(argv[1]);
        cellsPerDimension = make_int3( Nc, Nc, Nc );
        numberOfCells = Nc*Nc*Nc;
        
        initialNumberOfAtoms = atoi(argv[2]);
        numberOfAtoms = initialNumberOfAtoms;
        alpha = h_N / numberOfAtoms;
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
    double3 *d_evapPos;
    double3 *d_evapVel;
    
    cuDoubleComplex *d_psiUp;
    cuDoubleComplex *d_psiDn;
    
    double2 *d_localPopulations;
    
    double *d_sigvrmax;
    
    double time = 0.;
    double medianR;
    double Temp = Tinit;
    
    int2 *d_cellStartEnd;
    
    hbool_t *d_atomIsSpinUp;
    
    int *d_cellID;
    int *d_numberOfAtomsInCell;
    int *d_prefixScanNumberOfAtomsInCell;
    int *d_collisionCount;
    int *d_atomID;
    int *d_evapTag;
    int *d_newEnd;

    cudaCalloc( (void **)&d_pos, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_vel, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_acc, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_evapPos, numberOfAtoms, sizeof(double3) );
    cudaCalloc( (void **)&d_evapVel, numberOfAtoms, sizeof(double3) );
    
    cudaCalloc( (void **)&d_psiUp, numberOfAtoms, sizeof(cuDoubleComplex) );
    cudaCalloc( (void **)&d_psiDn, numberOfAtoms, sizeof(cuDoubleComplex) );
    
    cudaCalloc( (void **)&d_localPopulations, numberOfAtoms, sizeof(double2) );
    
    cudaCalloc( (void **)&d_sigvrmax, numberOfCells+1, sizeof(double) );
    
    cudaCalloc( (void **)&d_cellStartEnd, numberOfCells+1, sizeof(int2) );
    
    cudaCalloc( (void **)&d_atomIsSpinUp, numberOfAtoms, sizeof(hbool_t) );
    
    cudaCalloc( (void **)&d_cellID, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_numberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_prefixScanNumberOfAtomsInCell, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_collisionCount, numberOfCells+1, sizeof(int) );
    cudaCalloc( (void **)&d_atomID, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_evapTag, numberOfAtoms, sizeof(int) );
    cudaCalloc( (void **)&d_newEnd, 1, sizeof(int) );
    
    double3 *h_pos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_vel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_evapPos = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    double3 *h_evapVel = (double3*) calloc( numberOfAtoms, sizeof(double3) );
    
    cuDoubleComplex *h_psiUp = (cuDoubleComplex*) calloc( numberOfAtoms, sizeof(cuDoubleComplex) );
    cuDoubleComplex *h_psiDn = (cuDoubleComplex*) calloc( numberOfAtoms, sizeof(cuDoubleComplex) );
    
    hbool_t *h_atomIsSpinUp = (hbool_t*) calloc( numberOfAtoms, sizeof(hbool_t) );
    
    int *h_numberOfAtomsInCell = (int*) calloc( numberOfCells+1, sizeof(int) );
    int *h_collisionCount = (int*) calloc( numberOfCells+1, sizeof(int) );
	int *h_cellID = (int*) calloc( numberOfAtoms, sizeof(int) );
    int *h_atomID = (int*) calloc( numberOfAtoms, sizeof(int) );
    
    thrust::device_ptr<int> th_numberOfAtomsInCell = thrust::device_pointer_cast( d_numberOfAtomsInCell );
    thrust::device_ptr<int> th_prefixScanNumberOfAtomsInCell = thrust::device_pointer_cast( d_prefixScanNumberOfAtomsInCell );
    thrust::device_ptr<int> th_atomID = thrust::device_pointer_cast( d_atomID );
    thrust::device_ptr<int> th_evapTag = thrust::device_pointer_cast( d_evapTag );
    thrust::device_ptr<int> th_newEnd = thrust::device_pointer_cast( d_newEnd );
    
#pragma mark - Set up atom system
	
    dt = 1.0e-7;
    double tau = 128.*sqrt( h_mRb*pow( h_pi, 3 )*pow( h_kB*Temp,5 ) ) / ( h_sigma*h_N*pow( h_gs*h_muB*h_dBdz, 3 ) );
//    int loopsPerCollision = ceil( 0.1*tau / dt );
    int loopsPerCollision = 10;
    int collisionsPerPrint = ceil( finalTime / ( loopsPerCollision * numberOfPrints * dt ) );
//    int collisionsPerPrint = 1;
    
    printf("tau/10 = %g, lpc = %i, cpp = %i\n", tau/10., loopsPerCollision, collisionsPerPrint );
    
	copyConstantsToDevice<<<1,1>>>( dt );
	
    h_initRNG( d_rngStates, numberOfAtoms );
    
    h_generateInitialDist( d_pos,
                           d_vel,
                           d_acc,
                          d_psiUp,
                          d_psiDn,
                           numberOfAtoms,
                           Tinit,
                           d_rngStates,
                           d_atomID,
                          d_atomIsSpinUp );
    
    initSigvrmax( d_sigvrmax, numberOfCells );
    
    medianR = indexAtoms(d_pos,
                         d_cellID,
                         d_atomID,
                         cellsPerDimension,
                         numberOfAtoms );
    sortArrays(d_cellID,
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
    
    cudaMemcpy(h_evapPos,
               d_evapPos,
               numberOfAtoms*sizeof(double3),
               cudaMemcpyDeviceToHost );
    char evapPosDatasetName[] = "/atomData/evapPos";
    hdf5FileHandle hdf5handleEvapPos = createHDF5Handle(atomDims,
                                                        H5T_NATIVE_DOUBLE,
                                                        evapPosDatasetName );
    intialiseHDF5File(hdf5handleEvapPos,
                      filename );
    writeHDF5File(hdf5handleEvapPos,
                  filename,
                  h_evapPos );
    
    cudaMemcpy(h_evapVel,
               d_evapVel,
               numberOfAtoms*sizeof(double3),
               cudaMemcpyDeviceToHost );
    char evapVelDatasetName[] = "/atomData/evapVel";
    hdf5FileHandle hdf5handleEvapVel = createHDF5Handle(atomDims,
                                                        H5T_NATIVE_DOUBLE,
                                                        evapVelDatasetName );
    intialiseHDF5File(hdf5handleEvapVel,
                      filename );
    writeHDF5File(hdf5handleEvapVel,
                  filename,
                  h_evapVel );
    
    cudaMemcpy(h_psiUp,
               d_psiUp,
               numberOfAtoms*sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost );
    char psiUpDatasetName[] = "/atomData/psiUp";
    int3 psiDims = { numberOfAtoms, 2, 1 };
    hdf5FileHandle hdf5handlePsiUp = createHDF5Handle(psiDims,
                                                      H5T_NATIVE_DOUBLE,
                                                      psiUpDatasetName );
    intialiseHDF5File(hdf5handlePsiUp,
                      filename );
    writeHDF5File(hdf5handlePsiUp,
                  filename,
                  h_psiUp );
    
    cudaMemcpy(h_psiDn,
               d_psiDn,
               numberOfAtoms*sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost );
    char psiDnDatasetName[] = "/atomData/psiDn";
    hdf5FileHandle hdf5handlePsiDn = createHDF5Handle(psiDims,
                                                      H5T_NATIVE_DOUBLE,
                                                      psiDnDatasetName );
    intialiseHDF5File(hdf5handlePsiDn,
                      filename );
    writeHDF5File(hdf5handlePsiDn,
                  filename,
                  h_psiDn );
    
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
    
    cudaMemcpy( h_cellID,
                d_cellID,
                numberOfAtoms*sizeof(int),
                cudaMemcpyDeviceToHost );
    char cellIDDatasetName[] = "/atomData/cellID";
    int3 particleDims = { numberOfAtoms, 1, 1 };
    hdf5FileHandle hdf5handleCellID = createHDF5Handle( particleDims,
                                                        H5T_NATIVE_INT,
                                                        cellIDDatasetName );
    intialiseHDF5File( hdf5handleCellID,
                       filename );
    writeHDF5File( hdf5handleCellID,
                   filename,
                   h_cellID );
    
    cudaMemcpy(h_atomID,
               d_atomID,
               numberOfAtoms*sizeof(int),
               cudaMemcpyDeviceToHost );
    char atomIDDatasetName[] = "/atomData/atomID";
    hdf5FileHandle hdf5handleAtomID = createHDF5Handle(particleDims,
                                                       H5T_NATIVE_INT,
                                                       atomIDDatasetName );
    intialiseHDF5File( hdf5handleAtomID,
                      filename );
    writeHDF5File( hdf5handleAtomID,
                  filename,
                  h_atomID );
    
    cudaMemcpy(h_atomIsSpinUp,
               d_atomIsSpinUp,
               numberOfAtoms*sizeof(hbool_t),
               cudaMemcpyDeviceToHost );
    char atomIsSpinUpDatasetName[] = "/atomData/atomIsSpinUp";
    hdf5FileHandle hdf5handleAtomIsSpinUp = createHDF5Handle(particleDims,
                                                             H5T_NATIVE_HBOOL,
                                                             atomIsSpinUpDatasetName );
    intialiseHDF5File(hdf5handleAtomIsSpinUp,
                      filename );
    writeHDF5File(hdf5handleAtomIsSpinUp,
                  filename,
                  h_atomIsSpinUp );
    
#pragma mark - Main Loop
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                       &blockSize,
                                       (const void *) moveAtoms,
                                       0,
                                       sizeOfRNG );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    cudaDeviceGetAttribute(&numSMs,
                           cudaDevAttrMultiProcessorCount,
                           device);
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    printf("blocksize = %i, gridsize = %i\n", blockSize, gridSize);
    
    for (int i=0; i<numberOfPrints; i++)
    {
        for (int j=0; j<collisionsPerPrint; j++) {
            
#pragma mark Collide Atoms
            
            medianR = indexAtoms( d_pos,
                                 d_cellID,
                                 d_atomID,
                                 cellsPerDimension,
                                 numberOfAtoms );
            
            sortArrays( d_cellID,
                       d_atomID,
                       numberOfAtoms );
            
            deviceMemset<<<numberOfCells+1,1>>>( d_cellStartEnd,
                                                make_int2( -1, -1 ),
                                                numberOfCells + 1 );
            
            cellStartandEndKernel<<<gridSize,blockSize>>>(d_cellID,
                                                          d_atomID,
                                                          d_cellStartEnd,
                                                          initialNumberOfAtoms,
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
                                         d_atomID );
            
#pragma mark Evolve System
            
            getLocalPopulations<<<gridSize,blockSize>>>(d_pos,
                                                        d_psiUp,
                                                        d_psiDn,
                                                        d_localPopulations,
                                                        d_atomID,
                                                        numberOfAtoms );
            
            for (int k=0; k<loopsPerCollision; k++) {
                evolveWavefunction<<<gridSize,blockSize>>>(d_pos,
                                                           d_psiUp,
                                                           d_psiDn,
                                                           d_atomID,
                                                           numberOfAtoms );
                
                moveAtoms<<<gridSize,blockSize>>>(d_pos,
                                                  d_vel,
                                                  d_acc,
                                                  d_atomIsSpinUp,
                                                  d_atomID,
                                                  numberOfAtoms );
#pragma mark Evaporate Atoms
                Temp = calculateTemp(d_vel,
                                     d_atomID,
                                     numberOfAtoms );
                h_evaporationTag(d_pos,
                                 d_vel,
                                 d_evapPos,
                                 d_evapVel,
                                 d_psiUp,
                                 d_psiDn,
                                 d_atomID,
                                 d_evapTag,
                                 Temp,
                                 numberOfAtoms );
                
                cudaMemcpy( (void*)&d_Temp, (void*)&Temp, 1*sizeof(double), cudaMemcpyHostToDevice );
                th_newEnd = thrust::remove_if(th_atomID,
                                              th_atomID + numberOfAtoms,
                                              th_evapTag,
                                              thrust::identity<int>());
                
                numberOfAtoms = (int)(th_newEnd - th_atomID);
            }
            
            flipAtoms<<<gridSize,blockSize>>>(d_pos,
                                              d_vel,
                                              d_psiUp,
                                              d_psiDn,
                                              d_localPopulations,
                                              d_atomIsSpinUp,
                                              d_atomID,
                                              d_rngStates,
                                              numberOfAtoms );
            
            exponentialDecay<<<gridSize,blockSize>>>(d_pos,
                                                     d_psiUp,
                                                     d_psiDn,
                                                     d_atomID,
                                                     loopsPerCollision*dt,
                                                     numberOfAtoms );
            
            normalise<<<gridSize,blockSize>>>(d_psiUp,
                                              d_psiDn,
                                              d_atomID,
                                              numberOfAtoms );
            
            time += loopsPerCollision * dt;
        }
        
        cudaMemcpy( h_pos, d_pos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_vel, d_vel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_evapPos, d_evapPos, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_evapVel, d_evapVel, numberOfAtoms*sizeof(double3), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_psiUp, d_psiUp, numberOfAtoms*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_psiDn, d_psiDn, numberOfAtoms*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_numberOfAtomsInCell, d_numberOfAtomsInCell, (numberOfCells+1)*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_collisionCount, d_collisionCount, (numberOfCells+1)*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_cellID, d_cellID, numberOfAtoms*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy( h_atomID, d_atomID, numberOfAtoms*sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy(h_atomIsSpinUp, d_atomIsSpinUp, numberOfAtoms*sizeof(hbool_t), cudaMemcpyDeviceToHost );
        
        writeHDF5File( hdf5handlePos,
                      filename,
                      h_pos );
        writeHDF5File( hdf5handleVel,
                      filename,
                       h_vel );
        writeHDF5File( hdf5handleEvapPos,
                      filename,
                      h_evapPos );
        writeHDF5File( hdf5handleEvapVel,
                      filename,
                      h_evapVel );
        writeHDF5File(hdf5handlePsiUp,
                      filename,
                      h_psiUp );
        writeHDF5File(hdf5handlePsiDn,
                      filename,
                      h_psiDn );
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
        writeHDF5File( hdf5handleCellID,
                       filename,
                       h_cellID );
        writeHDF5File( hdf5handleAtomID,
                       filename,
                       h_atomID );
        
        writeHDF5File(hdf5handleAtomIsSpinUp,
                      filename,
                      h_atomIsSpinUp );
        
        printf("%%%i complete, t = %g s\n", i*100/numberOfPrints, time);
    }
    
    // insert code here...
    printf("\n");
    
    free( h_pos );
    free( h_vel );
    free( h_evapPos );
    free( h_evapVel );
    free( h_psiUp );
    free( h_psiDn );
    free( h_numberOfAtomsInCell );
	free( h_cellID );
    free( h_atomID );
    free( h_atomIsSpinUp );
    
    cudaFree( d_pos );
    cudaFree( d_vel );
    cudaFree( d_acc );
    cudaFree( d_evapPos );
    cudaFree( d_evapVel );
    cudaFree( d_psiUp );
    cudaFree( d_psiDn );
    cudaFree( d_sigvrmax );
    cudaFree( d_cellStartEnd );
    cudaFree( d_cellID );
    cudaFree( d_atomID );
    cudaFree( d_numberOfAtomsInCell );
    cudaFree( d_prefixScanNumberOfAtomsInCell );
    cudaFree( d_collisionCount );
    cudaFree( d_rngStates );
    cudaFree( d_evapTag );
    cudaFree( d_localPopulations );
    cudaFree( d_atomIsSpinUp );
    
    cudaDeviceReset();
    
    return 0;
}

