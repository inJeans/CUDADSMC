//
//  evaporation.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 4/10/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include "declareInitialSystemParameters.cuh"
#include "deviceSystemParameters.cuh"
#include "cudaHelpers.cuh"
#include "evaporation.cuh"

void evaporateAtoms( double3 *d_pos,
                     double3 *d_vel,
                     double3 *d_acc,
                     zomplex *d_psiU,
                     zomplex *d_psiD,
                     double2 *d_oldPops2,
                     hbool_t *d_isSpinUp,
                     int *d_cellID,
                     int *d_atomID,
                     double medianR,
                     int *numberOfAtoms )
{
    int *d_evapStencil;
    cudaCalloc( (void **)&d_evapStencil, numberOfAtoms[0], sizeof(int) );
    
    checkForEvapAtoms( d_pos, d_isSpinUp, medianR, d_evapStencil, numberOfAtoms[0] );
    
    int remainingAtoms = thrust::reduce( thrust::device, d_evapStencil, d_evapStencil + numberOfAtoms[0] );
    
    compactArrayd3( d_pos,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayd3( d_vel,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayd3( d_acc,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayZ ( d_psiU, d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayZ ( d_psiD, d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayd2( d_oldPops2,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayB ( d_isSpinUp,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayI ( d_cellID,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    compactArrayI ( d_atomID,  d_evapStencil, numberOfAtoms[0], remainingAtoms );
    
    numberOfAtoms[0] = remainingAtoms;
    
    cudaFree( d_evapStencil );
}

void checkForEvapAtoms( double3 *d_pos, hbool_t *d_isSpinUp, double medianR, int *d_evapStencil, int numberOfAtoms )
{
    int blockSize;
    int minGridSize;
    int gridSize;
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) d_checkForEvapAtoms,
                                        0,
                                        numberOfAtoms );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
    
    d_checkForEvapAtoms<<<gridSize,blockSize>>>( d_pos, d_isSpinUp, medianR, d_evapStencil, numberOfAtoms );
    
    return;
}

__global__ void d_checkForEvapAtoms( double3 *pos, hbool_t *isSpinUp, double medianR, int *evapStencil, int numberOfAtoms )
{
    for ( int atom = blockIdx.x * blockDim.x + threadIdx.x;
          atom < numberOfAtoms;
          atom += blockDim.x * gridDim.x)
    {
        bool isOutsideGrid = checkAtomGridPosition( pos[atom], medianR );
        
        if ( !isSpinUp[atom] && isOutsideGrid ) {
            evapStencil[atom] = 0;
        }
        else
        {
            evapStencil[atom] = 1;
        }
    }
    
    return;
}

__device__ bool checkAtomGridPosition( double3 pos, double medianR )
{
    bool isOutsideGrid = false;
    
    double gridMin =-d_meshWidth * medianR;
    double gridMax = d_meshWidth * medianR;
    
    if ( pos.x < gridMin || pos.x > gridMax ||
         pos.y < gridMin || pos.y > gridMax ||
         pos.z < gridMin || pos.z > gridMax )
    {
        isOutsideGrid = true;
    }
    
    return isOutsideGrid;
}

void compactArrayd3( double3 *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms )
{
    double3 *d_temp;
    cudaCalloc( (void **)&d_temp, numberOfAtoms, sizeof(double3) );
    
    thrust::copy_if( thrust::device, d_array, d_array + numberOfAtoms, d_evapStencil, d_temp, thrust::identity<int>() );
    thrust::copy( thrust::device, d_temp, d_temp + remainingAtoms, d_array );
    
    cudaFree( d_temp );
    
    return;
}

void compactArrayd2( double2 *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms )
{
    double2 *d_temp;
    cudaCalloc( (void **)&d_temp, numberOfAtoms, sizeof(double2) );
    
    thrust::copy_if( thrust::device, d_array, d_array + numberOfAtoms, d_evapStencil, d_temp, thrust::identity<int>() );
    thrust::copy( thrust::device, d_temp, d_temp + remainingAtoms, d_array );
    
    cudaFree( d_temp );
    
    return;
}

void compactArrayZ( zomplex *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms )
{
    zomplex *d_temp;
    cudaCalloc( (void **)&d_temp, numberOfAtoms, sizeof(zomplex) );
    
    thrust::copy_if( thrust::device, d_array, d_array + numberOfAtoms, d_evapStencil, d_temp, thrust::identity<int>() );
    thrust::copy( thrust::device, d_temp, d_temp + remainingAtoms, d_array );
    
    cudaFree( d_temp );
    
    return;
}

void compactArrayB( hbool_t *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms )
{
    hbool_t *d_temp;
    cudaCalloc( (void **)&d_temp, numberOfAtoms, sizeof(hbool_t) );
    
    thrust::copy_if( thrust::device, d_array, d_array + numberOfAtoms, d_evapStencil, d_temp, thrust::identity<int>() );
    thrust::copy( thrust::device, d_temp, d_temp + remainingAtoms, d_array );
    
    cudaFree( d_temp );
    
    return;
}

void compactArrayI( int *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms )
{
    int *d_temp;
    cudaCalloc( (void **)&d_temp, numberOfAtoms, sizeof(int) );
    
    thrust::copy_if( thrust::device, d_array, d_array + numberOfAtoms, d_evapStencil, d_temp, thrust::identity<int>() );
    thrust::copy( thrust::device, d_temp, d_temp + remainingAtoms, d_array );
    
    cudaFree( d_temp );
    
    return;
}