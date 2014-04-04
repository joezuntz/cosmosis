#!/usr/bin/env python
import sys
import desglue
import pydesglue
import numpy as np
import pylab
import copy
import os

params = {
"OMEGA_B": 0.04,
"OMEGA_M": 0.25,
"H0":      0.72,
"N_S":     0.97,
"A_S":     2.0e-9,
"OPT_TAU": 0.08,
}

data_file="data.fits"
if not os.path.exists(data_file):
	data = pydesglue.DesDataPackage.from_cosmo_params(params)
	camb=pydesglue.load_module("../../boltzmann/camb/camb_all.so","execute")
	h = data.to_new_fits_handle()
	print "Start CAMB"
	status = camb(h)
	print "camb status was:", status
	data = pydesglue.DesDataPackage.from_fits_handle(h)
	data.save_to_file("data.fits")
else:
	data = pydesglue.DesDataPackage.from_file(data_file)
	h = data.to_new_fits_handle()

halofit=pydesglue.load_module("../../boltzmann/halofit/halofit_interface.so","execute")
status = halofit(h)
print "halofit status was:", status
data = pydesglue.DesDataPackage.from_fits_handle(h)


def plot_p(data,section,iz=0,label=""):
	k=data.get_data(section,"K_H")
	z=data.get_data(section,"Z")
	P=data.get_data(section,"P_K")
	nk=data.get_param(section,"NK")
	nz=data.get_param(section,"NZ")
	P=P.reshape((nk,nz))
	k=k[::nz]
	#print k
	z=z[:nz]
	pylab.loglog(k,P[:,iz], label="%s"%(label), lw=3)


plot_p(data, "PK_NL",0, label="Halofit non-linear")
plot_p(data, "PK_LIN",0, label="Linear")
# plot_p(data, "PK_NL",100)
# plot_p(data, "PK_LIN",100)
pylab.legend(loc='lower left')
pylab.show()
	
	
