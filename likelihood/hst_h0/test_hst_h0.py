import pydesglue
import numpy as np

def like(h0):
	params = {
		'H0' : h0,
	}
	
	package = pydesglue.DesDataPackage.from_cosmo_params(params)
	handle = package.to_new_fits_handle()
	module = pydesglue.load_module("./hst_h0.py","execute")
	status = module(handle)
	assert status ==0, "Failed to run HST module"
	
	package = pydesglue.DesDataPackage.from_fits_handle(handle)
	pydesglue.free_fits(handle)
	like = package.get_param(pydesglue.section_names.likelihoods, "HST_LIKE")
	return like
	
	
if __name__== "__main__":
	for h0 in np.arange(0.4, 1.0, 0.01):
		print h0, like(h0)
