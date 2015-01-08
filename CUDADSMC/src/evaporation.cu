//
//  evaporation.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 1/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "vectorMath.cuh"
#include "evaporation.cuh"
#include "math.h"
#include "cudaHelpers.cuh"

#include "declareInitialSystemParameters.cuh"
#include "deviceSystemParameters.cuh"

void h_evaporationTag(double3 *d_pos,
                      double3 *d_vel,
                      double3 *d_evapPos,
                      double3 *d_evapVel,
                      hbool_t *d_atomIsSpinUp,
                      int     *d_atomID,
                      int     *d_evapTag,
                      double   Temp,
                      int      numberOfAtoms )
{
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                       &blockSize,
                                       (const void *) evaporationTag,
                                       0,
                                       numberOfAtoms );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    int numSMs;
    cudaDeviceGetAttribute(&numSMs,
                           cudaDevAttrMultiProcessorCount,
                           device);
    
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    evaporationTag<<<gridSize,blockSize>>>(d_pos,
                                           d_vel,
                                           d_evapPos,
                                           d_evapVel,
                                           d_atomIsSpinUp,
                                           d_atomID,
                                           d_evapTag,
                                           Temp,
                                           numberOfAtoms );
    
    return;
}

__global__ void evaporationTag(double3 *pos,
                               double3 *vel,
                               double3 *evapPos,
                               double3 *evapVel,
                               hbool_t *atomIsSpinUp,
                               int     *atomID,
                               int     *evapTag,
                               double   Temp,
                               int      numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        int l_atom = atomID[atom];
        double3 l_pos = pos[l_atom];
        double3 l_vel = vel[l_atom];
        
        if ( atomIsSpinUp[l_atom] ) {
            evapTag[atom] = 0;
        }
        else
        {
            evapTag[atom] = 1;
            evapPos[l_atom] = l_pos;
            evapVel[l_atom] = l_vel;
        }
    }
    
    return;
}

double calculateTemp(double3 *d_vel,
                     int *d_atomID,
                     int numberOfAtoms )
{
    double *d_speed2;
    cudaCalloc( (void **)&d_speed2, numberOfAtoms, sizeof(double) );
    
    h_calculateSpeed2(d_vel,
                      d_atomID,
                      d_speed2,
                      numberOfAtoms );
    
    double T  = h_mRb / 3. / h_kB * findMean(d_speed2,
                                             numberOfAtoms );

//    printf("The temperature is %fuK\n", T * 1.e6 );
    
    cudaFree( d_speed2 );
    
    return T;
}

void h_calculateSpeed2(double3 *d_vel,
                       int     *d_atomID,
                       double  *d_speed2,
                       int      numberOfAtoms )
{
    int blockSize;
    int gridSize;
    
#ifdef CUDA65
    int minGridSize;
    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize,
                                       &blockSize,
                                       (const void *) calculateSpeed2,
                                       0,
                                       numberOfAtoms );
    gridSize = (numberOfAtoms + blockSize - 1) / blockSize;
#else
    int device;
    cudaGetDevice ( &device );
    int numSMs;
    cudaDeviceGetAttribute(&numSMs,
                           cudaDevAttrMultiProcessorCount,
                           device);
    
    gridSize = 256*numSMs;
    blockSize = NUM_THREADS;
#endif
    
    calculateSpeed2<<<gridSize,blockSize>>>(d_vel,
                                            d_atomID,
                                            d_speed2,
                                            numberOfAtoms );
    
    return;
}

__global__ void calculateSpeed2(double3 *vel,
                                int     *atomID,
                                double  *speed2,
                                int      numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        speed2[atom] = dot( vel[atomID[atom]], vel[atomID[atom]] );
    }
    
    return;
}

double findMean( double *v, int N )
{
    thrust::device_ptr<double> th_v = thrust::device_pointer_cast( v );
    
    double sum = thrust::reduce( th_v, th_v + N );
    
    return sum / N;
}

__device__ double3 getMagneticFieldN( double3 pos )
{
    double3 B = getMagneticF( pos );
    
    double3 Bn = B / length( B );
    
    return Bn;
}

__device__ double3 getMagneticF( double3 pos )
{
    double3 B = d_dBdz * make_double3( 0.5*pos.x, 0.5*pos.y, -1.0*pos.z );
    
    return B;
}