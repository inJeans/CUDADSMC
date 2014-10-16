//
//  collisions.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "declareInitialSystemParameters.cuh"
#include "deviceSystemParameters.cuh"
#include "collisions.cuh"
#include "cudaHelpers.cuh"

#pragma mark - Indexing

double indexAtoms( double3 *d_pos, int *d_cellID, int3 cellsPerDimension )
{
    double *d_radius;
    cudaCalloc( (void **)&d_radius, numberOfAtoms, sizeof(double) );
    
	h_calculateRadius( d_pos,
                       d_radius,
                       numberOfAtoms );
    
    double medianR = findMedian( d_radius,
                                 numberOfAtoms );
    
    printf("The median radius is %f\n", medianR );
    
    h_findAtomIndex( d_pos,
                     d_cellID,
                     medianR,
                     numberOfAtoms,
                     cellsPerDimension );
    
    cudaFree( d_radius );
    
    return medianR;
}

void h_calculateRadius( double3 *d_pos, double *d_radius, int numberOfAtoms )
{
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) calculateRadius,
                                        0,
                                        sizeOfRNG );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    int numSMs;
    cudaDeviceGetAttribute( &numSMs,
                            cudaDevAttrMultiProcessorCount,
                            device);
    
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    calculateRadius<<<gridSize,blockSize>>>( d_pos,
                                             d_radius,
                                             numberOfAtoms );
    
    return;
}

__global__ void calculateRadius( double3 *pos, double *radius, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        radius[atom] = length( pos[atom] );
    }
    
    return;
}

double findMedian( double *v, int N )
{
    thrust::device_ptr<double> ptr = thrust::device_pointer_cast( v );
    
    thrust::sort( ptr,
                  ptr + N );
    
    double *d_median;
    cudaCalloc( (void **)&d_median, 1, sizeof(double) );
    
    getMedian<<<1,1>>>( v, d_median, N );
    
    double h_median;
    
    cudaMemcpy( (void *)&h_median, d_median, 1*sizeof(double), cudaMemcpyDeviceToHost );
    
    cudaFree( d_median );
    
    return h_median;
}

__global__ void getMedian( double *v, double *median, int N)
{
    if (N % 2 == 0) {
        median[0] = 0.5*(v[N/2-1] + v[N/2]);
    }
    else {
        median[0] = v[(N-1)/2];
    }
    
    return;
}

void h_findAtomIndex( double3 *d_pos, int *d_cellID, double medianR, int numberOfAtoms, int3 cellsPerDimension )
{
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) calculateRadius,
                                        0,
                                        sizeOfRNG );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    int numSMs;
    cudaDeviceGetAttribute( &numSMs,
                            cudaDevAttrMultiProcessorCount,
                            device);
    
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    findAtomIndex<<<gridSize,blockSize>>>( d_pos,
                                           d_cellID,
                                           medianR,
                                           numberOfAtoms,
                                           cellsPerDimension );
    
    return;
}

__global__ void findAtomIndex( double3 *pos, int *cellID, double medianR, int numberOfAtoms, int3 cellsPerDimension )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        double3 l_pos = pos[atom];
        
        double3 gridMin    = getGridMin( medianR );
        double3 cellLength = getCellLength( medianR,
                                            cellsPerDimension );
    
        int3 cellIndices = getCellIndices( l_pos,
                                           gridMin,
                                           cellLength );
		
        cellID[atom] = getCellID( cellIndices, cellsPerDimension );
    }
    
    return;
}

__device__ double3 getCellLength( double medianR, int3 cellsPerDimension )
{
    double3 cellLength = 2.0 * d_maxGridWidth / cellsPerDimension;
    
    return cellLength;
}

__device__ int3 getCellIndices( double3 pos, double3 gridMin, double3 cellLength )
{
    int3 index = { 0, 0, 0 };
    
    index.x = __double2int_rd ( (pos.x - gridMin.x) / cellLength.x );
    index.y = __double2int_rd ( (pos.y - gridMin.y) / cellLength.y );
    index.z = __double2int_rd ( (pos.z - gridMin.z) / cellLength.z );
	
    return index;
}

__device__ int getCellID( int3 index, int3 cellsPerDimension )
{
    int cellID = 0;
    
    if (index.x > -1 && index.x < cellsPerDimension.x && index.y > -1 && index.y < cellsPerDimension.y && index.z > -1 && index.z < cellsPerDimension.z) {
        cellID = index.z*cellsPerDimension.x*cellsPerDimension.y + index.y*cellsPerDimension.x + index.x;
    }
    else {
        cellID = cellsPerDimension.x * cellsPerDimension.y * cellsPerDimension.z;
    }
    
    return cellID;
}

__device__ double3 getGridMin( double medianR )
{
    double3 gridMin = -1.0 * d_maxGridWidth;
    
    return  gridMin;
}

__global__ void cellStartandEndKernel( int *cellID, int2 *cellStartEnd, int numberOfAtoms )
{
	for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        // Find the beginning of the cell
        if (atom == 0) {
            cellStartEnd[cellID[atom]].x = 0;
        }
        else if (cellID[atom] != cellID[atom-1]) {
            cellStartEnd[cellID[atom]].x = atom;
        }
        
        // Find the end of the cell
        if (atom == numberOfAtoms - 1) {
            cellStartEnd[cellID[atom]].y = numberOfAtoms-1;
        }
        else if (cellID[atom] != cellID[atom+1]) {
            cellStartEnd[cellID[atom]].y = atom;
        }
    }
    
    return;
}

__device__ void serialCellStartandEndKernel( int *cellID, int2 *cellStartEnd, int numberOfAtoms )
{
	for ( int atom = 0;
		  atom < numberOfAtoms;
		  atom++ )
	{
        // Find the beginning of the cell
        if (atom == 0) {
            cellStartEnd[cellID[atom]].x = 0;
        }
        else if (cellID[atom] != cellID[atom-1]) {
            cellStartEnd[cellID[atom]].x = atom;
        }
        
        // Find the end of the cell
        if (atom == numberOfAtoms - 1) {
            cellStartEnd[cellID[atom]].y = numberOfAtoms-1;
        }
        else if (cellID[atom] != cellID[atom+1]) {
            cellStartEnd[cellID[atom]].y = atom;
        }
    }
    
    return;
}

__global__ void findNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells )
{
    for ( int cell = blockIdx.x * blockDim.x + threadIdx.x;
		  cell < numberOfCells+1;
		  cell += blockDim.x * gridDim.x)
	{
		if (cellStartEnd[cell].x == -1)
		{
			numberOfAtomsInCell[cell] = 0;
		}
		else
		{
			numberOfAtomsInCell[cell] = cellStartEnd[cell].y - cellStartEnd[cell].x + 1;
		}
    }
    
    return;
}

__device__ void serialFindNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells )
{
    for (int cell = 0;
		 cell < numberOfCells;
		 cell++ )
	{
        numberOfAtomsInCell[cell] = cellStartEnd[cell].y - cellStartEnd[cell].x + 1;
    }
    
    return;
}

#pragma mark - Sorting

void sortArrays( double3 *d_pos,
                 double3 *d_vel,
                 double3 *d_acc,
                 int *d_cellID )
{
    thrust::device_ptr<double3> th_pos = thrust::device_pointer_cast( d_pos );
    thrust::device_ptr<double3> th_vel = thrust::device_pointer_cast( d_vel );
    thrust::device_ptr<double3> th_acc = thrust::device_pointer_cast( d_acc );
    
    thrust::device_ptr<int> th_cellID = thrust::device_pointer_cast( d_cellID );
    
    thrust::device_vector<int>  th_indices( numberOfAtoms );
    thrust::sequence( th_indices.begin(),
                      th_indices.end() );
    
    thrust::sort_by_key( th_cellID,
                         th_cellID + numberOfAtoms,
                         th_indices.begin() );
    
    double3 *d_sorted;
    cudaCalloc( (void **)&d_sorted, numberOfAtoms, sizeof(double3) );
    thrust::device_ptr<double3> th_sorted = thrust::device_pointer_cast( d_sorted );
    
    thrust::gather( th_indices.begin(),
                    th_indices.end(),
                    th_pos,
                    th_sorted );
    th_pos = th_sorted;

    thrust::gather( th_indices.begin(),
                    th_indices.end(),
                    th_vel,
                    th_sorted );
    th_vel = th_sorted;
    
    thrust::gather( th_indices.begin(),
                    th_indices.end(),
                    th_acc,
                    th_sorted );
    th_acc = th_sorted;
    
    cudaFree( d_sorted );
    
    return;
}

#pragma mark - Collisions

__global__ void collide( double3 *vel,
                         double  *sigvrmax,
                         int     *prefixScanNumberOfAtomsInCell,
                         int     *collisionCount,
                         double   medianR,
                         double   alpha,
                         int3     cellsPerDimension,
                         int      numberOfCells,
                         curandStatePhilox4_32_10_t *rngState,
                         int *cellID )
{
    for ( int cell = blockIdx.x * blockDim.x + threadIdx.x;
          cell < numberOfCells;
          cell += blockDim.x * gridDim.x)
    {
        int numberOfAtomsInCell = prefixScanNumberOfAtomsInCell[cell+1] - prefixScanNumberOfAtomsInCell[cell];
        
        if (numberOfAtomsInCell > 2) {
            double3 cellLength = getCellLength( medianR,
                                                cellsPerDimension );
            
            d_dt = 1.0e-6;
            d_loopsPerCollision = 0.005 / d_dt;
            
            double cellVolume = cellLength.x * cellLength.y * cellLength.z;
            double Mc = 0.5 * (numberOfAtomsInCell - 1) * numberOfAtomsInCell;
            double lambda = ceil( Mc * alpha * d_loopsPerCollision * d_dt * sigvrmax[cell] / cellVolume ) / Mc;
            int Ncol = Mc*lambda;
            
            double3 velcm, newVel, pointOnSphere;
            
            double crossSection = 8.*d_pi*d_a*d_a;
            double magVrel;
            double ProbCol;
            
            for ( int l_collision = 0;
                  l_collision < Ncol;
                  l_collision++ )
            {
                curandStatePhilox4_32_10_t l_rngState = rngState[cell];
                
                int2 collidingAtoms = {0,0};
                
                if (numberOfAtomsInCell == 2) {
                    collidingAtoms.x = prefixScanNumberOfAtomsInCell[cell] + 0;
                    collidingAtoms.y = prefixScanNumberOfAtomsInCell[cell] + 1;
                }
                else {
                    collidingAtoms = prefixScanNumberOfAtomsInCell[cell] + chooseCollidingAtoms( numberOfAtomsInCell, &l_rngState );
                }
                
                magVrel = calculateRelativeVelocity( vel, collidingAtoms );
                
                // Check if this is the more probable than current most probable.
                if (magVrel*crossSection > sigvrmax[cell]) {
                    sigvrmax[cell] = magVrel * crossSection;
                }
                
                ProbCol = alpha * d_loopsPerCollision * d_dt / cellVolume * magVrel * crossSection / lambda;
                
                // Collide with the collision probability.
                if ( ProbCol > curand_uniform_double ( &l_rngState ) ) {
                    // Find centre of mass velocities.
                    velcm = 0.5*(vel[collidingAtoms.x] + vel[collidingAtoms.y]);
                    
                    // Generate a random velocity on the unit sphere.
                    pointOnSphere = getRandomPointOnSphere( &l_rngState );
                    newVel = magVrel * pointOnSphere;
                    
                    vel[collidingAtoms.x] = velcm - 0.5 * newVel;
                    vel[collidingAtoms.y] = velcm + 0.5 * newVel;
//                    collisionCount[cell] += d_alpha;
                    collisionCount[cell]++;
                }
                
                rngState[cell] = l_rngState;
                
            }
        }
    }

    return;
}

__device__ int2 chooseCollidingAtoms( int numberOfAtomsInCell, curandStatePhilox4_32_10_t *rngState )
{
    int2 collidingAtoms = { 0, 0 };
    
    // Randomly choose particles in this cell to collide.
    while (collidingAtoms.x == collidingAtoms.y) {
        collidingAtoms = double2Toint2_rd( curand_uniform2_double ( &rngState[0] ) * (numberOfAtomsInCell-1) );
    }
    
    return collidingAtoms;
}

__device__ double calculateRelativeVelocity( double3 *vel, int2 collidingAtoms )
{
    double3 vRel = vel[collidingAtoms.x] - vel[collidingAtoms.y];
    double magVrel = sqrt(vRel.x*vRel.x + vRel.y*vRel.y + vRel.z*vRel.z);
    
    return magVrel;
}

__device__ double3 getRandomPointOnSphere( curandStatePhilox4_32_10_t *rngState )
{
    double2 r1 = curand_normal2_double ( &rngState[0] );
    double  r2 = curand_normal_double  ( &rngState[0] );
    
    double3 pointOnSphere = make_double3( r1.x, r1.y, r2 ) * rsqrt( r1.x*r1.x + r1.y*r1.y + r2*r2 );
    
    return pointOnSphere;
}