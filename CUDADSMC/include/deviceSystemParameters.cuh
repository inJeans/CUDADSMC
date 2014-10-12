//
//  deviceSystemParameters.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 13/09/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_deviceSystemParameters_cuh
#define CUDADSMC_deviceSystemParameters_cuh

__constant__ double d_gs   =  0.5;			// Gyromagnetic ratio
__constant__ double d_MF   = -1.0;			// Magnetic quantum number
__constant__ double d_muB  = 9.27400915e-24;	// Bohr magneton
__constant__ double d_mRb  = 1.443160648e-25;// 87Rb mass
__constant__ double d_pi   = 3.14159265;		// Pi
__constant__ double d_a    = 5.3e-9;			// Constant cross-section formula
__constant__ double d_kB   = 1.3806503e-23;	// Boltzmann's Constant
__constant__ double d_hbar = 1.05457148e-34;	// hbar

__device__ int3   d_cellsPerDimension = { 10, 10, 10 };
__device__ int    d_numberOfCells = 10*10*10;
__device__ double d_meshWidth = 2.5;
__device__ double3 d_maxGridWidth = { 0.01, 0.01, 0.01 };
__device__ double d_alpha = 1.0*1.e7 / 1.e5;
__device__ double d_dt;
__device__ double d_dBdz = 2.5;
__device__ double d_wavelength = 6.06012e-8;

__device__ int d_loopsPerCollision;	// loops per collision

#endif
