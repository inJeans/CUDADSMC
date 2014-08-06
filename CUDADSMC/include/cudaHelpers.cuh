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

int setMaxCUDADevice( void );
int getMaxCUDADevice( void );
int findMaxCUDADevice( int numberOfCUDADevices );

#endif
