#!/usr/bin/env python
import matplotlib

#Set some plot options
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["xtick.major.size"] = 8
matplotlib.rcParams["ytick.major.size"] = 8
matplotlib.rcParams["xtick.minor.size"] = 4
matplotlib.rcParams["ytick.minor.size"] = 4

#Load libraries
import pylab
import numpy as np

#Load in the cosmosis output files
log_like = np.loadtxt("demo12_output/evs/logphi.txt").T
m = np.loadtxt("demo12_output/evs/m.txt").T

#convert to like from log-like and normalize
like = np.exp(log_like)
like /= like.max()

#make the plot
pylab.xscale("log")
pylab.plot(m, like, lw=2) # mass in M_sun

#Add a vertical bar for the observation:
#XXMU J0044  Santos et al. 2011
mass=4.02E14*0.7
pylab.plot([mass,mass],[0.0,1.0],
	label="XXMU J0044 z = 1.579", linestyle='--', lw=2, color="red")

#Set the axis and display setup
pylab.xlabel("Maximum Mass ($M_\odot /h$)")
pylab.ylabel("Likelihood")
pylab.minorticks_on()
pylab.legend()
pylab.grid()

#Save the file
pylab.savefig("demo12_pdf.png")
