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

time = np.zeros((tres));
pos = np.zeros((ntrials,3,tres));
vel = np.zeros((ntrials,3,tres));

for i in range(1,10):
    for j in range(5,8):
        
        sourcefile = 'Tests/Motion/verlet-' + str(i) + '.e-' + str(j) + '.h5'

        f = h5py.File( sourcefile );

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
        
        dt[i+(j-5)*10] = i*pow(10.,-j);
        dE[i+(j-5)*10] = max( abs(Et - Et[0]) / Et[0] * 100 );


#p = np.polyfit( np.log(dt), np.log(dE), 1 );

#dEfit = np.exp(p[1])*pow(dt,p[0])

pl.loglog( dt, dE, 'o' )
pl.loglog( dt, dEfit )
pl.xlabel(r'$\Delta t$')
pl.ylabel(r'$|\Delta E|_{max}$')

pl.show()

	