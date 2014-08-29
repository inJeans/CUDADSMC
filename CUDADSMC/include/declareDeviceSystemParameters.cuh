//
//  declareDeviceSystemParameters.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 28/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_declareDeviceSystemParameters_cuh
#define CUDADSMC_declareDeviceSystemParameters_cuh

#pragma mark - Device globals

extern __constant__ double d_gs   =  0.5;			// Gyromagnetic ratio
extern __constant__ double d_MF   = -1.0;			// Magnetic quantum number
extern __constant__ double d_muB  = 9.27400915e-24;	// Bohr magneton
extern __constant__ double d_mRb  = 1.443160648e-25;// 87Rb mass
extern __constant__ double d_pi   = 3.14159265;		// Pi
extern __constant__ double d_a    = 5.3e-9;			// Constant cross-section formula
extern __constant__ double d_kB   = 1.3806503e-23;	// Boltzmann's Constant
extern __constant__ double d_hbar = 1.05457148e-34;	// hbar

extern __device__ int3   d_cellsPerDimension = { 10, 10, 10 };
extern __device__ int    d_numberOfCells = 10*10*10;
extern __device__ double d_Fn = 1.;
extern __device__ double d_meshWidth = 1.5;
extern __device__ double d_alpha = 1.e7 / 1.e4;
extern __device__ double d_dt = 1.0e-6;
extern __device__ double d_dBdz = 2.5;

extern __device__ int d_loopsPerCollision = 1000;	// loops per collision

#endif
