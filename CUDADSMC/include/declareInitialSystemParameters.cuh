//
//  declareInitialSystemParameters.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_declareInitialSystemParameters_cuh
#define CUDADSMC_declareInitialSystemParameters_cuh

#pragma mark -

extern int    numberOfAtoms;
extern int    numberOfCells;
extern int3   cellsPerDimension;
extern double dt;

#pragma mark Environmental Parameters
extern double Tinit;

#pragma mark Trap Parameters
extern double dBdz;

#pragma mark - Device globals

extern __device__ int3 d_cellsPerDimension = { 2, 2, 2 };
extern __device__ int  d_numberOfCells = 2*2*2;

#endif