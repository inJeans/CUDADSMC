//
//  cudaHelpers.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include "cudaHelpers.cuh"

#pragma mark - Set Maximum CUDA Device
int setMaxCUDADevice( void )
{
	int maxDevice = getMaxCUDADevice( );
	
	cudaSetDevice( maxDevice );
	
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( &deviceProp, maxDevice );
	
	printf("\n----------------------------------------\n");
	printf("\n Running on \"%s\"\n", deviceProp.name );
	printf("\n----------------------------------------\n\n"); 
	
	return maxDevice;
}

int getMaxCUDADevice( void )
{
	int numberOfCUDADevices, maxDevice;
	
	cudaGetDeviceCount( &numberOfCUDADevices );
	
	if ( numberOfCUDADevices > 1) {
		maxDevice = findMaxCUDADevice( numberOfCUDADevices );
	}
	else {
		maxDevice = 0;
	}
	
	return maxDevice;
}

int findMaxCUDADevice( int numberOfCUDADevices )
{
	int maxNumberOfMP = 0, maxDevice = 0;
	
	for (int device = 0; device < numberOfCUDADevices; device++) {
		cudaDeviceProp currentDeviceProp;
		cudaGetDeviceProperties( &currentDeviceProp, device );
		
		if ( maxNumberOfMP < currentDeviceProp.multiProcessorCount ) {
			maxNumberOfMP = currentDeviceProp.multiProcessorCount;
			maxDevice = device;
		}
	}
	
	return maxDevice;
}

void cudaSetMem( double *d_array, double value, int lengthOfArray )
{
    double *h_array = (double*) calloc( lengthOfArray, sizeof(double) );
    
    for ( int i=0;
          i<lengthOfArray;
          i++ )
    {
        h_array[i] = value;
    }
    
    cudaMemcpy( d_array, h_array, lengthOfArray*sizeof(double), cudaMemcpyHostToDevice );

    free( h_array );
    
    return;
}

__global__ void deviceMemset( double *d_array, double value, int lengthOfArray )
{
	for ( int element = blockIdx.x * blockDim.x + threadIdx.x;
		  element < lengthOfArray;
		  element += blockDim.x * gridDim.x)
	{
		d_array[element] = value;
	}
	return;
}

__global__ void deviceMemset( int2 *d_array, int2 value, int lengthOfArray )
{
	for ( int element = blockIdx.x * blockDim.x + threadIdx.x;
		 element < lengthOfArray;
		 element += blockDim.x * gridDim.x)
	{
		d_array[element] = value;
	}
	return;
}

__global__ void deviceMemset( int *d_array, int value, int lengthOfArray )
{
	for ( int element = blockIdx.x * blockDim.x + threadIdx.x;
		 element < lengthOfArray;
		 element += blockDim.x * gridDim.x)
	{
		d_array[element] = value;
	}
	return;
}
