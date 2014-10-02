//
//  spinEvolution.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 11/09/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_spinEvolution_cuh
#define CUDADSMC_spinEvolution_cuh

__global__ void unitaryEvolution( zomplex *psiU, zomplex *psiD, double2 *oldPops2, double3 *pos, double3 *vel, int numberOfAtoms );
__device__ double3 getMagneticField( double3 pos );
__device__ double2 getEigenStatePopulations( zomplex psiD, zomplex psiU, double3 Bn );
__global__ void projectSpins( zomplex *psiU, zomplex *psiD, double2 *oldPops2, double3 *pos, double3 *vel, hbool_t *isSpinUp, curandStatePhilox4_32_10_t *rngstate, int numberOfAtoms );
__global__ void exponentialDecay( zomplex *psiU, zomplex *psiD, double3 *pos, hbool_t *isSpinUp, int numberOfAtoms );
__global__ void normaliseWavefunction( zomplex *psiU, zomplex *psiD, int numberOfAtoms );
__device__ double3 updateTheAcc( double3 pos );

#endif
