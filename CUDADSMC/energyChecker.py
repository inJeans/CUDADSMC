import h5py
import numpy as np
import scipy.stats as stats
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

tres = 51;
ntrials = 1e4;
dt = 1e-6;

time = np.zeros((tres));
pos = np.zeros((ntrials,3,tres));
vel = np.zeros((ntrials,3,tres));
N = np.zeros((tres));
isSpinUp = np.zeros((ntrials,1,tres));

fpos = np.zeros((ntrials,3,tres));
fvel = np.zeros((ntrials,3,tres));

f = h5py.File('outputData.h5');

dset = f.require_dataset('atomData/simuatedTime',(1,1,tres),False,False);
dset.read_direct(time);

dset = f.require_dataset('atomData/positions',(ntrials,3,tres),False,False);
dset.read_direct(pos);

dset = f.require_dataset('atomData/velocities',(ntrials,3,tres),False,False);
dset.read_direct(vel);

dset = f.require_dataset('atomData/isSpinUp',(ntrials,1,tres),False,False);
dset.read_direct(isSpinUp);

dset = f.require_dataset('atomData/atomNumber',(1,1,tres),False,False);
dset.read_direct(N);

dset = f.require_dataset('atomData/flippedPos',(ntrials,3,tres),False,False);
dset.read_direct(fpos);

dset = f.require_dataset('atomData/flippedVel',(ntrials,3,tres),False,False);
dset.read_direct(fvel);

f.close()

num = np.sum( isSpinUp, 0 )
isSpinUp = 2*isSpinUp[:,0,:] - 1;

Ek = np.zeros((N.size,))
Ep = np.zeros((N.size,))
Et = np.zeros((N.size,))
Temp = np.zeros((N.size,))

Tx = np.zeros((N.size,))
Ty = np.zeros((N.size,))
Tz = np.zeros((N.size,))

vx = np.zeros((N.size,))
vy = np.zeros((N.size,))
vz = np.zeros((N.size,))

avx = np.zeros((N.size,))
avy = np.zeros((N.size,))
avz = np.zeros((N.size,))

for i in range(0,N.size):
    Ek[i] = np.sum( 0.5 * mRb * np.sum(vel[0:N[i],:,i]**2, 1), 0 ) / N[i] / kB * 1.e6
    Ep[i] = np.sum( isSpinUp[0:N[i],i]*0.5*gs*muB*dBdz*np.sqrt(pos[0:N[i],0,i]**2 + pos[0:N[i],1,i]**2 + 4.0*pos[0:N[i],2,i]**2 ), 0 ) / N[i] / kB * 1.e6
    Et[i] = Ek[i]+Ep[i]

    Temp[i] = 2./3. * np.sum( 0.5 * mRb * np.sum(vel[0:N[i],:,i]**2, 1), 0) / N[i] / kB * 1.e6

    Tx[i] = 2./3. * np.sum( 0.5 * mRb * vel[0:N[i],0,i]**2, 0) / N[i] / kB * 1.e6
    Ty[i] = 2./3. * np.sum( 0.5 * mRb * vel[0:N[i],1,i]**2, 0) / N[i] / kB * 1.e6
    Tz[i] = 2./3. * np.sum( 0.5 * mRb * vel[0:N[i],2,i]**2, 0) / N[i] / kB * 1.e6

    vx[i] = 2./3. * np.sum( 0.5 * mRb * vel[0:N[i],0,i], 0) / N[i] / kB * 1.e6
    vy[i] = 2./3. * np.sum( 0.5 * mRb * vel[0:N[i],1,i], 0) / N[i] / kB * 1.e6
    vz[i] = 2./3. * np.sum( 0.5 * mRb * vel[0:N[i],2,i], 0) / N[i] / kB * 1.e6

    avx[i] = np.sum( pos[0:N[i],0,i], 0) / N[i] * 1.e6
    avy[i] = np.sum( pos[0:N[i],1,i], 0) / N[i] * 1.e6
    avz[i] = np.sum( pos[0:N[i],2,i], 0) / N[i] * 1.e6

#    fr = np.sqrt( fpos[:,0,i]**2 + fpos[:,1,i]**2 + fpos[:,2,i]**2 )
#    fp = np.where( fr > 0. )
#
#    up = np.where( isSpinUp[0:N[i],i] > 0 )
#    dn = np.where( isSpinUp[0:N[i],i] < 0 )
#
#    pl.clf()
#    pl.plot( pos[up,0,i], pos[up,1,i], 'b.' )
#    pl.plot( fpos[fp,0,i], fpos[fp,1,i], 'r.' )
#    pl.axis([-0.0008, 0.0008, -0.0008, 0.0008])
#    pl.title( 't = ' + str(time[i]) + ' s')
#    pl.pause( 1 )

dE = max( abs(Et - Et[0]) / Et[0] * 100 )

pl.clf()
pl.plot(time,Ek, '-o')
pl.plot(time,Ep, '-o')
pl.plot(time,Et, '-o')
pl.xlabel('time (s)')
pl.ylabel('energy (uK)')
pl.title( r'$(\Delta E)_{max}$ = %.3g' % dE + r',  $\Delta t$ = %.3g' % dt )

if dE < 1.e-3:
    print bcolors.OKGREEN + "Motion integrator passed, dE = %%%.3g, dt = %.3g" % (dE,dt) + bcolors.ENDC
else:
    print bcolors.FAIL + "Motion integrator failed, dE = %%%.3g, dt = %.3g" % (dE,dt) + bcolors.ENDC

#pl.draw()
#figurename = './Tests/Motion/motionTest-%.3g' % dt + '.eps'
#pl.savefig( figurename )

#filename = './Tests/Motion/motionTest-%.3g' % dt + '.npy'
#file = open(filename, "w")

pl.figure(2)
pl.plot( time, N )
pl.xlabel('time (s)')
pl.ylabel('Atom Number')

pl.figure(3)
pl.plot( time, Temp )
pl.xlabel('time (s)')
pl.ylabel('Temperature (uK)')

pl.figure(4)
pl.plot( time, Tx, time, Ty, time, Tz )
pl.xlabel('time (s)')
pl.ylabel('Directional Temperature (uK)')

Eki = 0.5 * mRb * np.sum(vel[0:N[0],:,0]**2, 1) / kB * 1.e6
Epi = isSpinUp[0:N[0],0]*0.5*gs*muB*dBdz*np.sqrt(pos[0:N[0],0,0]**2 + pos[0:N[0],1,0]**2 + 4.0*pos[0:N[0],2,0]**2 ) / kB * 1.e6
Eti = Eki + Epi

nki, binski, patches = pl.hist(Eki,100)
nki = np.append([0], nki , axis=0)
npi, binspi, patches = pl.hist(Epi,100)
npi = np.append([0], npi , axis=0)
nti, binsti, patches = pl.hist(Eti,100)
nti = np.append([0], nti , axis=0)

Ekf = 0.5 * mRb * np.sum(vel[0:N[-1],:,-1]**2, 1) / kB * 1.e6
Epf = isSpinUp[0:N[-1],-1]*0.5*gs*muB*dBdz*np.sqrt(pos[0:N[-1],0,-1]**2 + pos[0:N[-1],1,-1]**2 + 4.0*pos[0:N[-1],2,-1]**2 ) / kB * 1.e6
Etf = Ekf + Epf

nkf, binskf, patches = pl.hist(Ekf,100)
nkf = np.append([0], nkf , axis=0)
npf, binspf, patches = pl.hist(Epf,100)
npf = np.append([0], npf , axis=0)
ntf, binstf, patches = pl.hist(Etf,100)
ntf = np.append([0], ntf , axis=0)

pl.figure(6)
pl.plot( binski, nki, binskf, nkf )
pl.xlabel(r'$E_k$ $(\mu K)$')

pl.figure(7)
pl.plot( binspi, npi, binspf, npf )
pl.xlabel(r'$E_p$ $(\mu K)$')

pl.figure(8)
pl.plot( binsti, nti, binstf, ntf )
pl.xlabel(r'$E_T$ $(\mu K)$')

#pl.figure(5)
#pl.plot( time, num[0,:], '-o' );
#pl.xlabel( 'time (s) ' )
#pl.ylabel( 'Number' )
#
#pl.figure(6)
#pl.plot( time, pos[0,0,:], time, pos[0,1,:], time, pos[0,2,:] );
#pl.xlabel( 'time   ' )
#pl.ylabel( 'pos' )


#pl.figure(4)
#pl.plot( time, vx, time, vy, time, vz )
#pl.xlabel('time (s)')
#pl.ylabel('Average velocity in each direction (uK)')
#
#pl.figure(5)
#pl.plot( time, avx, time, avy, time, avz )
#pl.xlabel('time (s)')
#pl.ylabel('Average postion in each direction (um)')

#si = np.sqrt( vel[:,0,0]**2 + vel[:,1,0]**2, vel[:,2,0]**2 )
#sf = np.sqrt( vel[:,0,N.size-1]**2 + vel[:,1,N.size-1]**2, vel[:,2,N.size-1]**2 )
#
#vxi = vel[:,0,0]
#vxf = vel[:,0,N.size-1]
#vyi = vel[:,1,0]
#vyf = vel[:,1,N.size-1]
#vzi = vel[:,2,0]
#vzf = vel[:,2,N.size-1]
#
#ni, binsi, patches = pl.hist(si, 100 )
#ni = np.append( [0], ni, axis=0);
#nf, binsf, patches = pl.hist(sf, 100 )
#nf = np.append( [0], nf, axis=0);
#
#nvxi, binsvxi, patches = pl.hist(vxi, 100 )
#nvxi = np.append( [0], nvxi, axis=0);
#nvxf, binsvxf, patches = pl.hist(vxf, 100 )
#nvxf = np.append( [0], nvxf, axis=0);
#nvyi, binsvyi, patches = pl.hist(vyi, 100 )
#nvyi = np.append( [0], nvyi, axis=0);
#nvyf, binsvyf, patches = pl.hist(vyf, 100 )
#nvyf = np.append( [0], nvyf, axis=0);
#nvzi, binsvzi, patches = pl.hist(vzi, 100 )
#nvzi = np.append( [0], nvzi, axis=0);
#nvzf, binsvzf, patches = pl.hist(vzf, 100 )
#nvzf = np.append( [0], nvzf, axis=0);
#
#print np.mean(si)
#print np.mean(sf)
#
#pl.figure(2)
#pl.plot(binsi, ni, binsf, nf)
#pl.xlabel(r'$s$')
#
#pl.figure(4)
#pl.plot(binsvxi, nvxi, binsvxf, nvxf)
#pl.xlabel(r'$v_x$')
#pl.figure(5)
#pl.plot(binsvyi, nvyi, binsvyf, nvyf)
#pl.xlabel(r'$v_y$')
#pl.figure(6)
#pl.plot(binsvzi, nvzi, binsvzf, nvzf)
#pl.xlabel(r'$v_z$')

pl.show()