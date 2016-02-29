import h5py
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

Nc = ( 5, 10 )
Nt = ( 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6 )
logNt = np.log10(Nt)
tauComp = np.array( [[42.2150, 43.1860, 43.3775, 43.6589, 43.6556, 43.5523, 43.5400],
                     [23.7750, 43.0250, 43.2810, 43.4815, 43.5560, 43.5623, 43.5569]] )

tau = 43.5568

err = (tauComp - tau) / tau * 100.

Ntv, Ncv = np.meshgrid( logNt, Nc )

fig = pl.figure()
ax = fig.gca(projection='3d')
ax.plot_surface( Ntv, Ncv, err, cmap=cm.coolwarm )
#ax.set_xscale('log')
pl.xlabel(r'$N_t$')
pl.ylabel(r'$N_c$')

pl.show()
