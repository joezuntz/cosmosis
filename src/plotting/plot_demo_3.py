#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")
import pylab
import numpy as np


data = np.loadtxt("demo3_output.txt").T

r_T = data[0]
log_like = data[1]

#convert to like from log-like and normalize
like = np.exp(log_like)
like /= like.max()

#make plot
pylab.plot(r_T, like)
pylab.xlabel("Tensor ratio r")
pylab.ylabel("Likelihood")
pylab.grid()

#Save
pylab.savefig("plots/demo3.png")
