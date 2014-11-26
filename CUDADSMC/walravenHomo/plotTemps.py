import numpy as np
import scipy.stats as stats
import pylab as pl
import matplotlib as mpl

name  = [r'$\,5.0$', r'$\,7.5$', r'$12.5$', r'$15.0$'];
temps = [5, 75, 125, 15];
color = ['r', 'b', 'g', 'm']
marker = ['^', 'v', '<', '>']

collrate = np.load('colRateHomo5uK.npy');
T        = np.load('tempHomo5uK.npy');
time     = np.load('timeHomo5uK.npy');

tauc = np.mean(collrate[10:-1]);
time = time * tauc * 4.

pl.plot( time[0:-1:4], T[0:-1:4]/T[0], 'ks')

for i in range(0,4):
    collrate = np.load('colRateHomo' + str(temps[i]) + 'uK.npy');
    T        = np.load('tempHomo' + str(temps[i]) + 'uK.npy');
    Tperturb = np.load('tempPerturbHomo' + str(temps[i]) + 'uK.npy');
    time     = np.load('timeHomo' + str(temps[i]) + 'uK.npy');

    tres = time.size

    tauc = np.mean(collrate[10:-1]);
    
    time = time * tauc * 4.
    
    slope, intercept, r_value, p_value, std_err = stats.linregress( time[0:0.25*tres], np.log(np.abs(Tperturb[0:0.25*tres] - Tperturb[-1])) )
    
    pl.plot( time[0:-1:4], Tperturb[0:-1:4]/T[0], marker[i], label=r'$T_p=$ ' + name[i] + r'$\,\mu\mathrm{K}$, $\tau / \tau_c =$ ' + '%.2f' % (-1/slope) + r'$\pm$ ' + '%.2f' % (1./slope * std_err/slope), markerfacecolor='none', markeredgewidth=2, markeredgecolor=color[i] )
    pl.plot( time, (Tperturb[-1] + np.sign(Tperturb[0] - T[0])*np.exp(intercept + slope*time))/T[0], color[i] )

    print "T = %s uK, Thermalisation in %f +- %f collisions" % (name[i], -1./slope, 1./slope * std_err/slope)

ax = pl.gca();

pl.axis([0, 14, 0.5, 1.5])

pl.ylabel(r'$T / T_\mathrm{bulk}(0)$', fontsize=24)
ylabels = ('0.50', '0.75', '1.00', '1.25', '1.50')
ax.set_yticks((0.5, 0.75, 1.0, 1.25, 1.5))
ax.set_yticklabels(ylabels, family='serif')

pl.xlabel(r'$t / \tau_c$', fontsize=24)
pl.tick_params(axis='both',labelsize=16)

legendProps = mpl.font_manager.FontProperties(family='serif',size=14)
pl.legend(prop=legendProps,numpoints=1)

pl.savefig('/Users/miMac/Documents/versionControlledFiles/miThesis/gfx/Thermalisation/walravenHomo.eps')
           
pl.show()