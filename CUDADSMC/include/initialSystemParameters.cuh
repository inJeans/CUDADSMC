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

int numberOfPrints = 25;
double finalTime = 0.5;
double dt = 1.e-6;

#pragma mark Physics Constants

double h_mRb   = 1.443160648e-25; // 87Rb mass;
double h_pi    = 3.14159265;		// Pi;
double h_a     = 5.3e-9;			// Constant cross-section formula;
double h_kB    = 1.3806503e-23;	// Boltzmann's Constant
double h_gs    =  0.5;			// Gyromagnetic ratio
double h_muB   = 9.27400915e-24;	// Bohr magneton
double h_sigma = 8.*h_pi*h_a*h_a;
int    h_N     = 3e6;

#pragma mark Environmental Parameters
double Tinit   = 20.0e-6;

#pragma mark Trap Parameters
double h_dBdz = 2.16;

#endif
