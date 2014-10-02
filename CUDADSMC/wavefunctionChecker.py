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

tres = 26;
ntrials = 1e4;
dt = 1e-6;

time = np.zeros((tres));
psiU = np.zeros((ntrials,2,tres));
psiD = np.zeros((ntrials,2,tres));
isSpinUp = np.zeros((ntrials,1,tres));

f = h5py.File('outputData.h5');

dset = f.require_dataset('atomData/simuatedTime',(1,1,tres),False,False);
dset.read_direct(time);

dset = f.require_dataset('atomData/psiU',(ntrials,2,tres),False,False);
dset.read_direct(psiU);

dset = f.require_dataset('atomData/psiD',(ntrials,2,tres),False,False);
dset.read_direct(psiD);

dset = f.require_dataset('atomData/isSpinUp',(ntrials,1,tres),False,False);
dset.read_direct(isSpinUp);
isSpinUp = 2*isSpinUp - 1;

f.close()

up = psiU[:,0,:] + 1j*psiU[:,1,:];
dn = psiD[:,0,:] + 1j*psiD[:,1,:];

norm = np.sum( up.conjugate()*up + dn.conjugate()*dn, 0 ).real / ntrials
dnorm = (norm - 1.) * 100.
dn = max(dnorm)

Pz = np.sum( isSpinUp, 0 ) / ntrials;

pl.clf()
pl.figure(1)
pl.plot(time,norm - 1.)
pl.xlabel('time (s)')
pl.ylabel(r'$\Delta \Psi$')
pl.title( r'$|\Delta \Psi|_{max}$ = %.3g' % dn + r',  $\Delta t$ = %.3g' % dt )

if dn < 1.e-3:
    print bcolors.OKGREEN + "Wavefunction integrator passed, dE = %%%.3g, dt = %.3g" % (dn,dt) + bcolors.ENDC
else:
    print bcolors.FAIL + "Wavefunction integrator failed, dE = %%%.3g, dt = %.3g" % (dn,dt) + bcolors.ENDC

#pl.draw()

pl.figure(2)
pl.plot(time,Pz[0,:])
pl.xlabel('time (s)')
pl.ylabel(r'$P_z$')
#pl.title( r'$|\Delta \Psi|_{max}$ = %.3g' % dn + r',  $\Delta t$ = %.3g' % dt )

#if dn < 1.e-3:
#    print bcolors.OKGREEN + "Wavefunction integrator passed, dE = %%%.3g, dt = %.3g" % (dn,dt) + bcolors.ENDC
#else:
#    print bcolors.FAIL + "Wavefunction integrator failed, dE = %%%.3g, dt = %.3g" % (dn,dt) + bcolors.ENDC

#pl.draw()

#figurename = './Tests/Motion/motionTest-%.3g' % dt + '.eps'
#pl.savefig( figurename )

#filename = './Tests/Motion/motionTest-%.3g' % dt + '.npy'
#file = open(filename, "w")

pl.show()