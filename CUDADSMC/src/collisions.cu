//
//  collisions.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "declareInitialSystemParameters.cuh"
#include "collisions.cuh"
#include "vectorMath.cuh"
#include "cudaHelpers.cuh"

#pragma mark - Indexing

void  indexAtoms( double3 *d_pos, int *d_cellID )
{
    float *d_radius;
    cudaCalloc( (void **)&d_radius, numberOfAtoms, sizeof(float) );
    
    int blockSize;
	int minGridSize;
	int gridSize;
	
	cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        calculateRadius,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("calculateRadius:     gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
	calculateRadius<<<gridSize,blockSize>>>( d_pos,
                                             d_radius,
                                             numberOfAtoms );
    
    float medianR = findMedian( d_radius,
                                numberOfAtoms );
    
    printf("The median radius is %f\n", medianR );
    
    cudaOccupancyMaxPotentialBlockSize( &minGridSize,
                                        &blockSize,
                                        findAtomIndex,
                                        0,
                                        numberOfAtoms );
	gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
	printf("findAtomIndex:       gridSize = %i, blockSize = %i\n", gridSize, blockSize);
    
    findAtomIndex<<<gridSize,blockSize>>>( d_pos, d_cellID, medianR, numberOfAtoms );
    
    cudaFree( d_radius );
    
    return;
}

__global__ void calculateRadius( double3 *pos, float *radius, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        radius[atom] = lengthf(pos[atom]);
    }
    
    return;
}

float findMedian( float *v, int N )
{
    thrust::device_ptr<float> ptr = thrust::device_pointer_cast( v );
    
    thrust::sort( ptr,
                  ptr + N );
    
    float *d_median;
    cudaCalloc( (void **)&d_median, 1, sizeof(float) );
    
    getMedian<<<1,1>>>( v, d_median, N );
    
    float h_median;
    
    cudaMemcpy( (void *)&h_median, d_median, 1*sizeof(float), cudaMemcpyDeviceToHost );
    
    cudaFree( d_median );
    
    return h_median;
}

__global__ void getMedian( float *v, float *median, int N)
{
    if (N % 2 == 0) {
        median[0] = 0.5*(v[N/2-1] + v[N/2]);
    }
    else {
        median[0] = v[(N-1)/2];
    }
    
    return;
}

__global__ void findAtomIndex( double3 *pos, int *cellID, float medianR, int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
		 atom < numberOfAtoms;
		 atom += blockDim.x * gridDim.x)
	{
        double3 l_pos = pos[atom];
    
        int3 indices = getCellIndices( l_pos,
                                       medianR );
        
        cellID[atom] = getCellID( indices );
    }
    
    return;
}

__device__ int3 getCellIndices( double3 pos, float medianR )
{
    int3 index = { 0, 0, 0 };
    
    float3 gridMin    = -1.5 * medianR * make_float3( 1., 1., 1. );
    float3 cellLength =  (float) 3.0 * medianR / d_cellsPerDimension;
    
    index.x = __float2int_rd ( (pos.x - gridMin.x) / cellLength.x );
    index.y = __float2int_rd ( (pos.y - gridMin.y) / cellLength.y );
    index.z = __float2int_rd ( (pos.z - gridMin.z) / cellLength.z );
    
    return index;
}

__device__ int getCellID( int3 index )
{
    int cellID = 0;
    
    if (index.x > -1 && index.x < d_cellsPerDimension.x && index.y > -1 && index.y < d_cellsPerDimension.y && index.z > -1 && index.z < d_cellsPerDimension.z) {
        cellID = index.z*d_cellsPerDimension.x*d_cellsPerDimension.y + index.y*d_cellsPerDimension.x + index.x;
    }
    else {
        cellID = d_cellsPerDimension.x * d_cellsPerDimension.y * d_cellsPerDimension.z;
    }
    
    return cellID;
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

__global__ void findNumberOfAtomsInCell( int2 *cellStartEnd, int *numberOfAtomsInCell, int numberOfCells )
{
    for (int cell = blockIdx.x * blockDim.x + threadIdx.x;
		 cell < numberOfCells+1;
		 cell += blockDim.x * gridDim.x)
	{
        printf("cell = %i\n", cell);
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

__global__ void collide( double3 *pos, double3 *vel, int *cellID, int *numberOfAtomsInCell, int numberOfCells )
{
    int cell   = blockIdx.x;
    int l_atom = threadIdx.x;
    int g_atom = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ double3 sh_pos[];
    if (l_atom < numberOfAtomsInCell[cell]) {
        sh_pos[l_atom] = pos[g_atom];
    }
    __syncthreads();
    
    return;
}