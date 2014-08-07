//
//  initialSystemParameters.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_initialSystemParameters_cuh
#define CUDADSMC_initialSystemParameters_cuh

#define NUM_THREADS 128

#pragma mark -

int    numberOfAtoms = 1e5;

#pragma mark Environmental Parameters
double Tinit   = 20.0e-6;

#pragma mark Trap Parameters
double dBdz = 2.5;

#endif
