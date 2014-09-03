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

dt = np.array( [5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6] );
dE = np.array( [2.74e-5, 4.19e-5, 4.55e-5, 3.79e-5, 0.000152, 9.53e-5, 0.000259, 0.000779, 0.00182, 0.00276] ) * 1e-2;

p = np.polyfit( np.log(dt), np.log(dE), 1 );

dEfit = np.exp(p[1])*pow(dt,p[0])

pl.loglog( dt, dE, 'o' )
pl.loglog( dt, dEfit )
pl.xlabel(r'$\Delta t$')
pl.ylabel(r'$|\Delta E|_{max}$')

pl.show()

	