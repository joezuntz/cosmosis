import sys
def usage():
	sys.stderr.write("Syntax:  %s  number_samples model_name\n(Suggest a few thousand samples and either 'eifler' or 'dodelson' for the model)\n"  % (sys.argv[0]))

model_name = "dodelson_test1"

try:
	number_samples = int(sys.argv[1])
except IndexError:
	usage()
	sys.exit(1)

import pydesglue
import dodelson_shear_model

import pymc
import numpy as np
import os
import cPickle




model = dodelson_shear_model.make_model()
params = [model['omega_m'], model['h0'], model['sigma8']]
#Pick and approximate covariance matrix
#estimated_covmat = np.diag([0.05**2, 0.02**2, 0.05**2])
estimated_covmat = np.array([[.0038,0.,-.0086],[0.,.02**2,0.],[-0.0086,0.,.02]])



	
database_file = model_name + "_chains"

#Set up the MCMC
mcmc=pymc.MCMC(model, db='txt', dbname=database_file, verbose=2)
mcmc.use_step_method(pymc.AdaptiveMetropolis, params, cov=estimated_covmat, delay=100, interval=100)
#Run the chain
mcmc.sample(number_samples)


mcmc.save_state()
mcmc.db.commit()
mcmc.db.close()
