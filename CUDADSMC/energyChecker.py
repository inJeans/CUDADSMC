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
muB  = 9.27400915e-24;
mRb  = 1.443160648e-25;
pi   = 3.14159265;
kB   = 1.3806503e-23;
hbar = 1.05457148e-34;
T    = 20.e-6;
dBdz = 2.5;

tres = 101;
ntrials = 1e4;
dt = 5e-6;

time = np.zeros((tres));
pos = np.zeros((ntrials,3,tres));
vel = np.zeros((ntrials,3,tres));

f = h5py.File('motionTest.h5');

dset = f.require_dataset('atomData/simuatedTime',(1,1,tres),False,False);
dset.read_direct(time);

dset = f.require_dataset('atomData/positions',(ntrials,3,tres),False,False);
dset.read_direct(pos);

dset = f.require_dataset('atomData/velocities',(ntrials,3,tres),False,False);
dset.read_direct(vel);

f.close()

Ek = np.sum( 0.5 * mRb * np.sum(vel**2, 1), 0 ) / ntrials / kB * 1.e6
Ep = np.sum( 0.5*gs*muB*dBdz*np.sqrt(pos[:,0,:]**2 + pos[:,1,:]**2 + 4.0*pos[:,2,:]**2 ), 0 ) / ntrials / kB * 1.e6
Et = Ek+Ep

dE = max( abs(Et - Et[0]) / Et[0] * 100 )

pl.clf()
pl.plot(time,Ek)
pl.plot(time,Ep)
pl.plot(time,Et)
pl.xlabel('time (s)')
pl.ylabel('energy (uK)')
pl.title( r'$(\Delta E)_{max}$ = %.3g' % dE + r',  $\Delta t$ = %.3g' % dt )

if dE < 1.e-3:
    print bcolors.OKGREEN + "Motion integrator passed, dE = %%%.3g, dt = %.3g" % (dE,dt) + bcolors.ENDC
else:
    print bcolors.FAIL + "Motion integrator failed, dE = %%%.3g, dt = %.3g" % (dE,dt) + bcolors.ENDC

pl.draw()
figurename = './Tests/Motion/motionTest-%.3g' % dt + '.eps'
pl.savefig( figurename )

filename = './Tests/Motion/motionTest-%.3g' % dt + '.npy'
file = open(filename, "w")
np.save( file, Ek )
np.save( file, Ep )
np.save( file, Et )
np.save( file, time )

pl.show()