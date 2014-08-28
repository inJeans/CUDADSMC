import h5py
import numpy as np
import pylab as pl
import os

# Set some physical constants
gs   = -0.5 * -1.0;
muB  = 9.27400915e-24;
mRb  = 1.443160648e-25;
pi   = 3.14159265;
kB   = 1.3806503e-23;
hbar = 1.05457148e-34;
T    = 20.e-6;
dBdz = 2.5;

tres = 101;
ntrials = 1e4;

pos = np.zeros((ntrials,3,tres));
vel = np.zeros((ntrials,3,tres));

f = h5py.File('outputData.h5');

dset = f.require_dataset('atomData/positions',(ntrials,3,tres),False,False);
dset.read_direct(pos);

dset = f.require_dataset('atomData/velocities',(ntrials,3,tres),False,False);
dset.read_direct(vel);

f.close()

Ek = np.sum( 0.5 * mRb * np.sum(vel**2, 1), 0 ) / ntrials / kB * 1.e6
Ep = np.sum( 0.5*gs*muB*dBdz*np.sqrt(pos[:,0,:]**2 + pos[:,1,:]**2 + 4.0*pos[:,2,:]**2 ), 0 ) / ntrials / kB * 1.e6