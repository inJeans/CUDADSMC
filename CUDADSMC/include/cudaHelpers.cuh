//
//  cudaHelpers.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_cudaHelpers_cuh
#define CUDADSMC_cudaHelpers_cuh

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define cudaCalloc(A, B, C) \
do { \
cudaError_t __cudaCalloc_err = cudaMalloc(A, (B)*C); \
if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, (B)*C); \
} while (0)

int setMaxCUDADevice( void );
int getMaxCUDADevice( void );
int findMaxCUDADevice( int numberOfCUDADevices );

#endif
