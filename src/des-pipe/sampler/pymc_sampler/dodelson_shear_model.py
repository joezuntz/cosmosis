import sys

import ctypes
import desglue
import pydesglue
import pyfits
import os
import pymc
import numpy as np
import pydesglue

shear_code = pydesglue.load_module("../../shear/dodelson_shear/dodelson_cl.py","execute")
likelihood_code = pydesglue.load_module("../../likelihood/dodelson_likelihood/dodelson_cl_likelihood.py","execute")


def make_model():
	#Helper function to work around an irritation in pymc.
	def get_value(p):
		if isinstance(p,float):
			return p
		elif isinstance(p,np.ndarray):
			if p.ndim==0:
				return float(p)
			else:
				return p[0]
		raise ValueError("Unknown type %r  (%r)" % (type(p), p))

		
	#Set up the parameters.  The starting values were found previously.  The limits are a little arbitrary.
	#Uniform priors are mostly not *too* bad for these.
	omega_m = pymc.Uniform("Omega_m", lower=0.1, upper=1.0,  value=0.26484236122)
	sigma8 =  pymc.Uniform("sigma_8", lower=0.5, upper=1.5,  value=0.800675400635)

	#Use the HST H0 prior on H0.
	h0 =      pymc.Normal("h0", mu=0.71, tau=0.02**-2,  value=0.71)



	@pymc.data
	@pymc.stochastic(verbose=1)
	def shear(omega_m=omega_m, h0=h0, sigma8=sigma8, value=0.0):
		#Check for negative values - for each of these parameters
		#that would indicate a bad sample.
		for p in [omega_m, h0, sigma8]:
			if p<0: return -np.inf
		#Package up the parameters to be passed to the codes
		data = pydesglue.DesDataPackage()
		section = pydesglue.section_names.cosmological_parameters
		data.add_section(section)
		data.set_param(section,"Omega_m",get_value(omega_m))
		data.set_param(section,"h0",get_value(h0))
		data.set_param(section,"h_0",get_value(h0))
		data.set_param(section,"sigma_8",get_value(sigma8))
		#Convert the package to an internal FITS file
		fits_handle = data.to_new_fits_handle()
		#Run the shear code to add shear power spectra to the data and the likelihood code to get the chi-squared
		result = shear_code(fits_handle)
		if result:
			print "Failed shear."
			desglue.free_fits(fits_handle)		
			return -np.inf
		result = likelihood_code(fits_handle)
		if result:
			print "Failed like."
			desglue.free_fits(fits_handle)		
			return -np.inf
		
		
		#Read the likelihood results back from the FITS file
		data = pydesglue.DesDataPackage.from_fits_handle(fits_handle)
		section = pydesglue.section_names.likelihoods
		like = data.get_param(section,"WL_LIKE")
		#Free the fits data.
		desglue.free_fits(fits_handle)		
		return like
		
	return locals()
