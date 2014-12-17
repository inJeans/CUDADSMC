//
//  wavefunction.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 8/12/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include "wavefunction.cuh"

__global__ void evolveWavefunction(double3 *pos,
                                   cuDoubleComplex *psiUp,
                                   cuDoubleComplex *psiDn,
                                   int *atomID,
                                   int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        int l_atom = atomID[atom];
        double3 l_pos = pos[l_atom];
        cuDoubleComplex l_psiUp = psiUp[l_atom];
        cuDoubleComplex l_psiDn = psiDn[l_atom];
        
        psiUp[l_atom] = updatePsiUp(l_pos,
                                    l_psiUp,
                                    l_psiDn );
        psiDn[l_atom] = updatePsiDn(l_pos,
                                    l_psiUp,
                                    l_psiDn );
        
    }
    
    return;
}


__device__ cuDoubleComplex updatePsiUp(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn )
{
    double3 Bn = getMagFieldN( pos );
    double  B  = getAbsB( pos );
    double  d_dt = 1.e-7;
    
    double theta = 0.5 * d_gs * d_muB * B * d_dt / d_hbar;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    
    cuDoubleComplex newPsiUp = make_cuDoubleComplex( psiUp.x*cosTheta + ( Bn.x*psiDn.y - Bn.y*psiDn.x + Bn.z*psiUp.y)*sinTheta,
                                                     psiUp.y*cosTheta + (-Bn.x*psiDn.x - Bn.y*psiDn.y - Bn.z*psiUp.x)*sinTheta );
    
    return newPsiUp;
}

__device__ cuDoubleComplex updatePsiDn(double3 pos,
                                       cuDoubleComplex psiUp,
                                       cuDoubleComplex psiDn )
{
    double3 Bn = getMagFieldN( pos );
    double  B  = getAbsB( pos );
    double  d_dt = 1.e-7;
    
    double theta = 0.5 * d_gs * d_muB * B * d_dt / d_hbar;
    double sinTheta = sin(theta);
    double cosTheta = cos(theta);
    
    cuDoubleComplex newPsiDn = make_cuDoubleComplex( psiDn.x*cosTheta + ( Bn.x*psiUp.y + Bn.y*psiUp.x - Bn.z*psiDn.y)*sinTheta,
                                                     psiDn.y*cosTheta + (-Bn.x*psiUp.x + Bn.y*psiUp.y + Bn.z*psiDn.x)*sinTheta );
    
    return newPsiDn;
}

__global__ void getLocalPopulations(double3 *pos,
                                    cuDoubleComplex *psiUp,
                                    cuDoubleComplex *psiDn,
                                    double2 *localPopulations,
                                    int *atomID,
                                    int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        int l_atom = atomID[atom];
        double3 l_pos = pos[l_atom];
        cuDoubleComplex l_psiUp = psiUp[l_atom];
        cuDoubleComplex l_psiDn = psiDn[l_atom];
        
        localPopulations[l_atom] = projectLocalPopulations(l_pos,
                                                           l_psiUp,
                                                           l_psiDn );
    }
    
    return;
}

__global__ void flipAtoms(double3 *pos,
                          double3 *vel,
                          cuDoubleComplex *psiUp,
                          cuDoubleComplex *psiDn,
                          double2 *localPopulations,
                          hbool_t *atomIsSpinUp,
                          int *atomID,
                          curandState_t *rngState,
                          int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        int l_atom = atomID[atom];
        double3 l_pos = pos[l_atom];
        double3 l_vel = vel[l_atom];
        cuDoubleComplex l_psiUp = psiUp[l_atom];
        cuDoubleComplex l_psiDn = psiDn[l_atom];
        double2 l_localPops = localPopulations[l_atom];
        hbool_t l_atomIsSpinUp = atomIsSpinUp[l_atom];
        
        double2 newLocalPopulations = projectLocalPopulations(l_pos,
                                                              l_psiUp,
                                                              l_psiDn );
        
        double magB = getAbsB( l_pos );
        
        double Pflip = 0.0;
        double dEp = 0.0;
        
        if (l_atomIsSpinUp) {
            Pflip = (newLocalPopulations.y - l_localPops.y) / l_localPops.x;
            dEp = 1.0 * d_gs * d_muB * magB;
        }
        else
        {
            Pflip = (newLocalPopulations.x - l_localPops.x) / l_localPops.y;
            dEp =-1.0 * d_gs * d_muB * magB;
        }
        
        double Ek = 0.5 * d_mRb * dot( l_vel, l_vel );
        
        if (Pflip > curand_uniform_double ( &rngState[l_atom] ) && Ek + dEp > 0) {
            atomIsSpinUp[l_atom] = !l_atomIsSpinUp;
            
            double magVel = length( l_vel );
            double3 Veln = l_vel / magVel;
            
            vel[l_atom] = sqrt( 2.0*( Ek + dEp ) / d_mRb ) * Veln;
        }
        
    }
    
    return;
}

__device__ double2 projectLocalPopulations(double3 pos,
                                           cuDoubleComplex psiUp,
                                           cuDoubleComplex psiDn )
{
    double2 localPops = make_double2( 0.0, 0.0 );
    
    double3 Bn = getMagFieldN( pos );
    
    localPops.x = 0.5 + Bn.x * ( psiUp.x*psiDn.x + psiUp.y*psiDn.y )
                      - Bn.y * ( psiUp.y*psiDn.x - psiUp.x*psiDn.y )
                      - Bn.z * ( 0.5 - psiUp.x*psiUp.x  - psiUp.y*psiUp.y );
   
    localPops.y = 0.5 - Bn.x * ( psiUp.x*psiDn.x + psiUp.y*psiDn.y )
                      + Bn.y * ( psiUp.y*psiDn.x - psiUp.x*psiDn.y )
                      + Bn.z * ( 0.5 - psiUp.x*psiUp.x  - psiUp.y*psiUp.y );
    
    return localPops;
}

__global__ void exponentialDecay(double3 *pos,
                                 cuDoubleComplex *psiUp,
                                 cuDoubleComplex *psiDn,
                                 int *atomID,
                                 double dt,
                                 int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        int l_atom = atomID[atom];
        double3 l_pos = pos[l_atom];
        cuDoubleComplex l_psiUp = psiUp[l_atom];
        cuDoubleComplex l_psiDn = psiDn[l_atom];
        
        double3 Bn = getMagFieldN( l_pos );
        double tau = calculateTau( l_pos );
        
        double oneminusexp = 1. - exp(-dt/tau);
        double oneplusexp = 1. + exp(-dt/tau);
        
        psiUp[l_atom] = 0.5 * make_cuDoubleComplex( (Bn.x*l_psiDn.x + Bn.y*l_psiDn.y + Bn.z*l_psiUp.x)*oneminusexp + l_psiUp.x*oneplusexp,
                                                    (Bn.x*l_psiDn.y - Bn.y*l_psiDn.x + Bn.z*l_psiUp.y)*oneminusexp + l_psiUp.y*oneplusexp );
        psiDn[l_atom] = 0.5 * make_cuDoubleComplex( (Bn.x*l_psiUp.x - Bn.y*l_psiUp.y - Bn.z*l_psiDn.x)*oneminusexp + l_psiDn.x*oneplusexp,
                                                    (Bn.x*l_psiUp.y + Bn.y*l_psiUp.x - Bn.z*l_psiDn.y)*oneminusexp + l_psiDn.y*oneplusexp );
    }
    
    return;
}

__device__ double calculateTau( double3 pos )
{
    double tau = 0.;
    double sigma = 8.*d_pi*d_a*d_a;
    
    double relAccel = 2.0 * length( getAcc( pos ) );
    
    double tauPos = 1.163 * sqrt( sigma / relAccel );
    double tauVel = sqrt( 2. / d_pi ) * d_hbar / ( d_mRb * sigma * relAccel );
    
    tau = pow( 1 / (tauPos*tauPos*tauPos) + 1 / (tauVel*tauVel*tauVel), -1./3. );
    
    return tau;
}

__device__ double3 getAcc( double3 pos )
{
    double3 accel = make_double3( 0., 0., 0. );
    
    double3 Bn = getMagFieldN( pos );
    double3 dBdx = getBdx( pos );
    double3 dBdy = getBdy( pos );
    double3 dBdz = getBdz( pos );
    
    double potential = -0.5 * d_gs * d_muB / d_mRb;
    
    accel.x = potential * dot( dBdx, Bn );
    accel.y = potential * dot( dBdy, Bn );
    accel.z = potential * dot( dBdz, Bn );
    
    return accel;
}

__device__ double3 getMagFieldN( double3 pos )
{
    double3 B = getBField( pos );
    
    double3 Bn = B / length( B );
    
    return Bn;
}

__device__ double getAbsB( double3 pos )
{
    double3 B = getBField( pos );
    
    return length( B );
}

__device__ double3 getBField( double3 pos )
{
    double3 B = d_B0     * make_double3( 0., 0., 1. ) +
                d_dBdx   * make_double3( pos.x, -pos.y, 0. ) +
         0.5 *  d_d2Bdx2 * make_double3( -pos.x*pos.z, -pos.y*pos.z, pos.z*pos.z - 0.5*(pos.x*pos.x+pos.y*pos.y) );
    
    return B;
}

__device__ double3 getBdx( double3 pos )
{
    return make_double3( d_dBdx - 0.5*d_d2Bdx2*pos.z, 0.0, - 0.5*d_d2Bdx2*pos.x );
}

__device__ double3 getBdy( double3 pos )
{
    return make_double3( 0.0, -d_dBdx - 0.5*d_d2Bdx2*pos.z, - 0.5*d_d2Bdx2*pos.y );
}

__device__ double3 getBdz( double3 pos )
{
    return make_double3( -0.5*d_d2Bdx2*pos.x, -0.5*d_d2Bdx2*pos.y, d_d2Bdx2*pos.z );
}

__global__ void normalise(cuDoubleComplex *psiUp,
                          cuDoubleComplex *psiDn,
                          int *atomID,
                          int numberOfAtoms )
{
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x;
         atom < numberOfAtoms;
         atom += blockDim.x * gridDim.x)
    {
        int l_atom = atomID[atom];
        cuDoubleComplex l_psiUp = psiUp[l_atom];
        cuDoubleComplex l_psiDn = psiDn[l_atom];
        
        double N = sqrt( l_psiUp.x*l_psiUp.x + l_psiUp.y*l_psiUp.y + l_psiDn.x*l_psiDn.x + l_psiDn.y*l_psiDn.y );
        
        psiUp[l_atom] = l_psiUp / N;
        psiDn[l_atom] = l_psiDn / N;
    }
    
    return;
}