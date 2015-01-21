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
dBdr = 8746.;

tres = 101;
ntrials = 1e6;
dt = 1e-6;

time = np.zeros((tres));
pos = np.zeros((ntrials,3,tres));
vel = np.zeros((ntrials,3,tres));
isPerturb = np.zeros((ntrials,1,tres));
N = np.zeros((tres));
collisionCount = np.zeros((10**3+1,1,tres));

f = h5py.File('/Users/miMac/Documents/versionControlledFiles/miThesis/Code/HomoWalraven/13auK.h5')
#f = h5py.File('outputData.h5');

dset = f.require_dataset('atomData/simuatedTime',(1,1,tres),False,False);
dset.read_direct(time);

dset = f.require_dataset('atomData/positions',(ntrials,3,tres),False,False);
dset.read_direct(pos);

dset = f.require_dataset('atomData/velocities',(ntrials,3,tres),False,False);
dset.read_direct(vel);

dset = f.require_dataset('atomData/isPerturb',(ntrials,1,tres),False,False);
dset.read_direct(isPerturb);

dset = f.require_dataset('atomData/atomNumber',(1,1,tres),False,False);
dset.read_direct(N);

#dset = f.require_dataset('atomData/collisionCount',(10**3+1,1,tres),False,False);
#dset.read_direct(collisionCount);

f.close()

totalColl = np.sum( collisionCount, 0 )[0,:];
#colls = np.cumsum( totalColl ) / N[0];
#time = colls;

#time = time * 24.21;

Ek = np.zeros((N.size,))
Ep = np.zeros((N.size,))
Et = np.zeros((N.size,))
Temp = np.zeros((N.size,))
Tperturb = np.zeros((N.size,))

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
    kinetic = 0.5 * mRb * np.sum(vel[0:N[i],:,i]**2, 1)
    n = np.where( np.isfinite(kinetic) )
    Ek[i] = np.sum( kinetic[n], 0 ) / N[i] / kB * 1.e6
    radius = pos[0:N[i],0,i]**2 + pos[0:N[i],1,i]**2 + pos[0:N[i],2,i]**2
    Ep[i] = np.sum( 0.5*gs*muB*dBdr*radius[n], 0 ) / N[i] / kB * 1.e6
    Et[i] = Ek[i] + Ep[i]

    Temp[i] = 2./3. * np.sum( kinetic[n], 0) / N[i] / kB * 1.e6
    
    kineticPerturb = isPerturb[0:N[i],0,i] * 0.5 * mRb * np.sum(vel[0:N[i],:,i]**2, 1)
    Tperturb[i] = 2./3. * np.sum( kineticPerturb[n], 0) / (0.01*ntrials) / kB * 1.e6

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

f = h5py.File('/Users/miMac/Documents/versionControlledFiles/miThesis/Code/HomoWalraven/13buK.h5')

dset = f.require_dataset('atomData/simuatedTime',(1,1,tres),False,False);
dset.read_direct(time);

dset = f.require_dataset('atomData/positions',(ntrials,3,tres),False,False);
dset.read_direct(pos);

dset = f.require_dataset('atomData/velocities',(ntrials,3,tres),False,False);
dset.read_direct(vel);

dset = f.require_dataset('atomData/isPerturb',(ntrials,1,tres),False,False);
dset.read_direct(isPerturb);

dset = f.require_dataset('atomData/atomNumber',(1,1,tres),False,False);
dset.read_direct(N);

f.close()

Tperturb2 = np.zeros((N.size,))

for i in range(0,N.size):
    kinetic = 0.5 * mRb * np.sum(vel[0:N[i],:,i]**2, 1)
    n = np.where( np.isfinite(kinetic) )
    
    kineticPerturb = isPerturb[0:N[i],0,i] * 0.5 * mRb * np.sum(vel[0:N[i],:,i]**2, 1)
    Tperturb2[i] = 2./3. * np.sum( kineticPerturb[n], 0) / (0.01*ntrials) / kB * 1.e6

Tperturb = (Tperturb + Tperturb2) / 2.;

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

fit = np.polyfit(time[0:0.04*tres], np.log(np.abs(Temp[-1] - Tperturb[0:0.04*tres])),1)

print "The thermalisation time is", -fit[0]
print "Thermalisation in %f collisions", (24.2027066667/-fit[0])*4

pl.figure(3)
pl.plot( time, Temp, 's', time, Tperturb, '^', time, Temp[-1] + np.exp(fit[1] + fit[0]*time), 'r' )
pl.xlabel('time (s)')
pl.ylabel('Temperature (uK)')

pl.figure(4)
pl.plot( time, Tx, time, Ty, time, Tz )
pl.xlabel('time (s)')
pl.ylabel('Directional Temperature (uK)')

#pl.figure(5)
#
#Eki = 0.5 * mRb * np.sum(vel[0:N[0],:,0]**2, 1) / kB * 1.e6
#Eki = Eki[np.isfinite(Eki)]
#Epi = np.sqrt(pos[0:N[0],0,0]**2 + pos[0:N[0],1,0]**2 + pos[0:N[0],2,0]**2 )
#Epi = Epi[np.isfinite(Epi)]
#Eti = Eki
#Li  = np.cross( vel[0:N[0],:,0], pos[0:N[0],:,0])
#Li  = Li[np.isfinite(Li)]
#
#nki, binski, patches = pl.hist(Eki,100)
#nki = np.append([0], nki , axis=0)
#npi, binspi, patches = pl.hist(Epi,100)
#npi = np.append([0], npi , axis=0)
#nti, binsti, patches = pl.hist(Eti,100)
#nti = np.append([0], nti , axis=0)
#nli, binsli, patches = pl.hist(Li,100)
#nli = np.append([0], nli , axis=0)
#
#Ekf = 0.5 * mRb * np.sum(vel[0:N[-1],:,-1]**2, 1) / kB * 1.e6
#Ekf = Ekf[np.isfinite(Ekf)]
#Epf = np.sqrt(pos[0:N[-1],0,-1]**2 + pos[0:N[-1],1,-1]**2 + pos[0:N[-1],2,-1]**2 )
#Epf = Epf[np.isfinite(Epf)]
#Etf = Ekf
#Lf  = np.cross( vel[0:N[-1],:,-1], pos[0:N[-1],:,-1])
#Lf  = Lf[np.isfinite(Lf)]
#
#nkf, binskf, patches = pl.hist(Ekf,100)
#nkf = np.append([0], nkf , axis=0)
#npf, binspf, patches = pl.hist(Epf,100)
#npf = np.append([0], npf , axis=0)
#ntf, binstf, patches = pl.hist(Etf,100)
#ntf = np.append([0], ntf , axis=0)
#nlf, binslf, patches = pl.hist(Lf,100)
#nlf = np.append([0], nlf , axis=0)
#
#pl.figure(6)
#pl.plot( binski, nki, binskf, nkf )
#pl.xlabel(r'$E_k$ $(\mu K)$')
#
#pl.figure(7)
#pl.plot( binspi, npi, binspf, npf )
#pl.xlabel(r'$E_p$ $(\mu K)$')
#
#pl.figure(8)
#pl.plot( binsti, nti, binstf, ntf )
#pl.xlabel(r'$E_T$ $(\mu K)$')
#
#pl.figure(9)
#pl.plot( binsli, nli, binslf, nlf )
#pl.xlabel(r'$L$')
#
#pl.figure(10)
#
#xi = pos[0:N[0],0,0]
#yi = pos[0:N[0],1,0]
#zi = pos[0:N[0],2,0]
#
#nxi, binsxi, patches = pl.hist(xi,100)
#nxi = np.append([0], nxi , axis=0)
#nyi, binsyi, patches = pl.hist(yi,100)
#nyi = np.append([0], nyi , axis=0)
#nzi, binszi, patches = pl.hist(zi,100)
#nzi = np.append([0], nzi , axis=0)
#
#xf = pos[0:N[-1],0,-1]
#yf = pos[0:N[-1],1,-1]
#zf = pos[0:N[-1],2,-1]
#
#nxf, binsxf, patches = pl.hist(xf,100)
#nxf = np.append([0], nxf , axis=0)
#nyf, binsyf, patches = pl.hist(yf,100)
#nyf = np.append([0], nyf , axis=0)
#nzf, binszf, patches = pl.hist(zf,100)
#nzf = np.append([0], nzf , axis=0)
#
#pl.figure(11)
#pl.plot( binsxi, nxi, binsxf, nxf )
#pl.xlabel(r'$x$')
#pl.figure(12)
#pl.plot( binsyi, nyi, binsyf, nyf )
#pl.xlabel(r'$y$')
#pl.figure(13)
#pl.plot( binszi, nzi, binszf, nzf )
#pl.xlabel(r'$z$')

pl.show()