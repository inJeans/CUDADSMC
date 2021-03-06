//
//  evaporation.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 10/11/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_evaporation_cuh
#define CUDADSMC_evaporation_cuh

void h_evaporationTag(double3 *d_pos,
                      double3 *d_vel,
                      double3 *d_evapPos,
                      double3 *d_evapVel,
                      cuDoubleComplex *d_psiUp,
                      cuDoubleComplex *d_psiDn,
                      int     *d_atomID,
                      int     *d_evapTag,
                      double   Temp,
                      int      numberOfAtoms );

__global__ void evaporationTag(double3 *pos,
                               double3 *vel,
                               double3 *evapPos,
                               double3 *evapVel,
                               cuDoubleComplex *psiUp,
                               cuDoubleComplex *psiDn,
                               int     *atomID,
                               int     *evapTag,
                               double   Temp,
                               int      numberOfAtoms );

double calculateTemp(double3 *d_vel,
                     int     *d_atomID,
                     int numberOfAtoms );
void h_calculateSpeed2(double3 *d_vel,
                       int     *d_atomID,
                       double  *d_speed2,
                       int      numberOfAtoms );
__global__ void calculateSpeed2(double3 *vel,
                                int     *atomID,
                                double  *speed2,
                                int      numberOfAtoms );

double findMean( double *v, int N );

__device__ double3 getMagneticFieldN( double3 pos );
__device__ double3 getMagneticF( double3 pos );

#endif
