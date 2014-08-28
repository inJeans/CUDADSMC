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
#include "declareDeviceSystemParameters.cuh"
#include "collisions.cuh"
#include "vectorMath.cuh"
#include "cudaHelpers.cuh"

#pragma mark - Indexing

double indexAtoms( double3 *d_pos, int *d_cellID )
{
    double *d_radius;
    cudaCalloc( (void **)&d_radius, numberOfAtoms, sizeof(double) );
    
    int blockSize;
	int minGridSize;
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) calculateRadius,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("calculateRadius:     gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
	calculateRadius<<<gridSize,blockSize>>>( d_pos,
                                             d_radius,
                                             numberOfAtoms );
    
    double medianR = findMedian( d_radius,
                                numberOfAtoms );
    
    printf("The median radius is %f\n", medianR );
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        (const void *) findAtomIndex,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("findAtomIndex:       gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    findAtomIndex<<<gridSize,blockSize>>>( d_pos, d_cellID, medianR, numberOfAtoms );
    
    cudaFree( d_radius );
    
    return medianR;
}

__global__ void calculateRadius( double3 *pos, double *radius, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        radius[atom] = lengthf(pos[atom]);
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

__global__ void findAtomIndex( double3 *pos, int *cellID, double medianR, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        double3 l_pos = pos[atom];
        
        double3 gridMin    = getGridMin( medianR );
        double3 cellLength = (double) 2.0 * d_meshWidth * medianR / d_cellsPerDimension;
    
        int3 cellIndices = getCellIndices( l_pos,
                                           gridMin,
                                           cellLength );
		
        cellID[atom] = getCellID( cellIndices, d_cellsPerDimension );
    }
    
    return;
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
    double3 gridMin = -d_meshWidth * medianR * make_double3( 1., 1., 1. );
    
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
	for (int atom = 0;
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
    for (int cell = blockIdx.x * blockDim.x + threadIdx.x;
		 cell < numberOfCells+1;
		 cell += blockDim.x * gridDim.x)
	{
        numberOfAtomsInCell[cell] = cellStartEnd[cell].y - cellStartEnd[cell].x + 1;
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

void sortArrays( double3 *d_pos, double3 *d_vel, double3 *d_acc, int *d_cellID )
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
                    th_sorted);
    th_vel = th_sorted;
    
    thrust::gather( th_indices.begin(),
                    th_indices.end(),
                    th_acc,
                    th_sorted);
    th_acc = th_sorted;
    
    cudaFree( d_sorted );
    
    return;
}

#pragma mark - Collisions

__global__ void collide( double3 *pos,
                         double3 *vel,
                         double  *sigvrmax,
                         int     *prefixScanNumberOfAtomsInCell,
                         double   medianR,
                         int      numberOfCells,
                         curandStatePhilox4_32_10_t *rngState )
{
    int cell   = blockIdx.x;
    int numberOfAtomsInCell = prefixScanNumberOfAtomsInCell[cell+1] - prefixScanNumberOfAtomsInCell[cell];
    int g_atom = 0;
    
    double3 cellLength    = (double) 2.0 * d_meshWidth * medianR / d_cellsPerDimension;
    
    __shared__ double3 sh_pos[MAXATOMS];
    __shared__ double3 sh_vel[MAXATOMS];
    __syncthreads();
    
    if (numberOfAtomsInCell > MAXATOMS) {
        numberOfAtomsInCell = MAXATOMS;
    }
    
    for ( int l_atom = threadIdx.x;
          l_atom < numberOfAtomsInCell;
		  l_atom += blockDim.x )
    {
        g_atom = prefixScanNumberOfAtomsInCell[cell] + l_atom;
        
        sh_pos[l_atom] = pos[g_atom];
        sh_vel[l_atom] = vel[g_atom];
    }
    __syncthreads();
    
    double cellVolume = cellLength.x * cellLength.y * cellLength.z;
    int Mc = __double2int_ru( 0.5 * d_alpha * (numberOfAtomsInCell - 1) * numberOfAtomsInCell * d_dt * sigvrmax[cell] / cellVolume );
    
    double3 vRel;
    int2 collidingAtoms;
    
    double crossSection = 8.*d_pi*d_a*d_a;
    double VR;
    
    for ( int l_collision = threadIdx.x;
          l_collision < Mc;
          l_collision += blockDim.x )
    {
        int g_collisionId = l_collision + cell*blockDim.x;
        curandStatePhilox4_32_10_t l_rngState = rngState[g_collisionId];
        
        collidingAtoms.x = 0;
        collidingAtoms.y = 0;
        
        if (numberOfAtomsInCell == 2) {
            collidingAtoms.x = 0;
            collidingAtoms.y = 1;
        }
        else {
            collidingAtoms = chooseCollidingAtoms( numberOfAtomsInCell, &l_rngState );
        }
        
        // Calculate the atoms' relative speed.
        vRel.x = vel[collidingAtoms.x].x - vel[collidingAtoms.y].x;
        vRel.y = vel[collidingAtoms.x].y - vel[collidingAtoms.y].y;
        vRel.z = vel[collidingAtoms.x].z - vel[collidingAtoms.y].z;
        VR     = sqrt(vRel.x*vRel.x + vRel.y*vRel.y + vRel.z*vRel.z);
        
        // Check if this is the more probable than current most probable.
        if (VR*crossSection > sigvrmax[cell]) {
            printf("Making it bigger %g -> %g\n", sigvrmax[cell], VR * crossSection);
            sigvrmax[cell] = VR * crossSection;
        }
        
        rngState[g_collisionId] = l_rngState;
    }
    
    __syncthreads();
    
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