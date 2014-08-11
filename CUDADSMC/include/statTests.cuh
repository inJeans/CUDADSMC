//
//  statTests.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 9/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_statTests_cuh
#define CUDADSMC_statTests_cuh

void shapiroWilk( double *data, int N );
__global__ void calcm( double *m, int N );
__device__ double norminv( double p, double mean, double stddev );

#endif
