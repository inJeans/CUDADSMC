//
//  initialSystemParameters.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_initialSystemParameters_cuh
#define CUDADSMC_initialSystemParameters_cuh

#pragma mark -

int3   cellsPerDimension = {5, 5, 5};
int    numberOfCells = cellsPerDimension.x*cellsPerDimension.y*cellsPerDimension.z;
int    numberOfAtoms = 1e3;
double alpha = 1e6 / numberOfAtoms;
double dt = 1.0e-6;

int numberOfPrints = 50;
int loopsPerCollision = 0.0007 / dt;

#pragma mark Physics Constants

double h_mRb = 1.443160648e-25; // 87Rb mass;
double h_pi  = 3.14159265;		// Pi;
double h_a   = 5.3e-9;			// Constant cross-section formula;
double h_kB  = 1.3806503e-23;	// Boltzmann's Constant

#pragma mark Environmental Parameters
double Tinit   = 20.0e-6;

#pragma mark Trap Parameters
double dBdz = 2.5;

#endif
