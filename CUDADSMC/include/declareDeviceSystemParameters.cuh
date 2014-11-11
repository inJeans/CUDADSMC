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

extern __constant__ double d_gs;			// Gyromagnetic ratio
extern __constant__ double d_MF;			// Magnetic quantum number
extern __constant__ double d_muB;	// Bohr magneton
extern __constant__ double d_mRb;// 87Rb mass
extern __constant__ double d_pi;		// Pi
extern __constant__ double d_a;			// Constant cross-section formula
extern __constant__ double d_kB;	// Boltzmann's Constant
extern __constant__ double d_hbar;	// hbar

extern __device__ double d_meshWidth;
extern __device__ double3 d_maxGridWidth;
extern __device__ double d_alpha;
extern __device__ double d_dt;
extern __device__ double d_dBdr;
extern __device__ double d_wavelength;

extern __device__ int d_loopsPerCollision;	// loops per collision
extern __device__ double d_Temp;

#endif
