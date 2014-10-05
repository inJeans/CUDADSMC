//
//  evaporation.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 4/10/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef __CUDADSMC__evaporation__cuh
#define __CUDADSMC__evaporation__cuh

#include "vectorMath.cuh"
#include "hdf5.h"

void evaporateAtoms( double3 *d_pos,
                    double3 *d_vel,
                    double3 *d_acc,
                    zomplex *d_psiU,
                    zomplex *d_psiD,
                    double2 *d_oldPops2,
                    hbool_t *d_isSpinUp,
                    int *d_cellID,
                    int *d_atomID,
                    double medianR,
                    int *numberOfAtoms );
void checkForEvapAtoms( double3 *d_pos, hbool_t *d_isSpinUp, double medianR, int *d_evapStencil, int numberOfAtoms );
__global__ void d_checkForEvapAtoms( double3 *pos, hbool_t *isSpinUp, double medianR, int *evapStencil, int numberOfAtoms );
__device__ bool checkAtomGridPosition( double3 pos, double medianR );
void compactArrayd3( double3 *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms );
void compactArrayd2( double2 *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms );
void compactArrayZ ( zomplex *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms );
void compactArrayB ( hbool_t *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms );
void compactArrayI ( int *d_array, int *d_evapStencil, int numberOfAtoms, int remainingAtoms );

#endif /* defined(__CUDADSMC__evaporation__cuh) */
