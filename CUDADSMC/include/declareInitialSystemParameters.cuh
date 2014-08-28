//
//  declareInitialSystemParameters.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 19/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_declareInitialSystemParameters_cuh
#define CUDADSMC_declareInitialSystemParameters_cuh

#define MAXATOMS 55
#define MAXSUBCELLS 64

#pragma mark -

extern int    numberOfAtoms;
extern int    numberOfCells;
extern int3   cellsPerDimension;
extern double dt;
extern double T;

#pragma mark Physical Constants

extern double h_mRb;
extern double h_pi;
extern double h_a;
extern double h_kB;

#pragma mark Environmental Parameters
extern double Tinit;

#pragma mark Trap Parameters
extern double dBdz;

#endif
