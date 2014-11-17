#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import pylab
import numpy as np


data = np.loadtxt("demo_output_massfunction/evs/logphi.txt").T
log_like = data
data = np.loadtxt("demo_output_massfunction/evs/m.txt").T
m = data

#convert to like from log-like and normalize
like = np.exp(log_like)
like /= like.max()

#make plot
pylab.xscale("log")
pylab.plot(m/0.70799558006015664, like) # mass in M_sun
pylab.xlabel("Max Mass (M_sun)")
pylab.ylabel("Likelihood")

#XXMU J0044  Santos et al. 2011
pylab.plot([4.02E14,4.02E14],[0.0,1.0],label="XXMU J0044 z = 1.579",linestyle='-.',color="red")
pylab.legend()
pylab.grid()

#Save
pylab.savefig("maxmass.png")
