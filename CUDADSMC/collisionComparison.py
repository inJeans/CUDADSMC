import h5py
import numpy as np
import pylab as pl
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[35m'
    FAIL = '\033[31m'
    ENDC = '\033[0m'

# Set some physical constants
gs   = -0.5 * -1.0;
a    = 5.3e-9;
muB  = 9.27400915e-24;
mRb  = 1.443160648e-25;
pi   = 3.14159265;
kB   = 1.3806503e-23;
hbar = 1.05457148e-34;
T    = 20.e-6;
dBdz = 2.5;

tres = 51;
nAtoms = 1e10;
nCells = 10**3 + 1;
dt = 5e-6;

vtherm = np.sqrt( 8.*kB*T / (pi*mRb) );
sigma = 8. * pi * a**2;
V = (2*0.001)**3
n0 = nAtoms / V
collisionRate = n0 * vtherm * sigma / np.sqrt(2.)

print "Analytic collision rate = ", collisionRate

tau   = np.zeros((13,10));
sigma = np.zeros((13,10));

for i in range(1,14):
    for j in range(1,11):
        
        nCells = j**3 + 1;
        
        time = np.zeros((tres));
        collisionCount = np.zeros((nCells,1,tres));
        N = np.zeros((tres));
        atomCount = np.zeros((nCells,1,tres))
        
        f = h5py.File('homogeneousBox-' + str(j) + '-' + str(1000*2**i) + '.h5');
        
        dset = f.require_dataset('atomData/simuatedTime',(1,1,tres),False,False);
        dset.read_direct(time);
        
        dset = f.require_dataset('atomData/collisionCount',(nCells,1,tres),False,False);
        dset.read_direct(collisionCount);
        
        dset = f.require_dataset('atomData/atomCount',(nCells,1,tres),False,False);
        dset.read_direct(atomCount);
        
        dset = f.require_dataset('atomData/atomNumber',(1,1,tres),False,False);
        dset.read_direct(N);
        
        f.close()
        
        totalColl = np.sum( collisionCount, 0 )[0,:];
        
        collRate = np.gradient( totalColl, time[1]-time[0] ) / N ;

        tau[i,j]   = np.mean(collRate[10:-1])
        sigma[i,j] = np.std (collRate[10:-1])

np.save( 'tau.npy', tau );
np.save( 'sigma.npy', sigma );
