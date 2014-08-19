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
extern double dt;

#pragma mark Environmental Parameters
extern double Tinit;

#pragma mark Trap Parameters
extern double dBdz;

#endif
