#!/usr/bin/env python
"""
This sampler program runs a simple Metropolis-Hastings parameter
constraint program on a model that is specifed in some parameter files.

It is generic, and flexibly loads both a collection of parameters to vary (and
the ranges and covariance estimate) and the physical model and data set which
they are used in.

The code uses the PyMC library to run the MCMC, and so it requires that as a 
dependency.

There are two parameter files used to specify a run. The first is the ini file.
An example of this is in params.ini, and comments in that example describe what
you can specify in that file.  Its main job is to describe the sequence of
calculation modules that will be run in the pipeline.  It also contains some
parameters that are specific to this sampler, including a covariance matrix
filename, which will be used for the initial proposal.

The other is the values file.  The name of this file is specified in the first
ini file.  An example is in values.ini.  This file specifies the parameters
that are to be fed into the pipeline - their names, min and max values, and
starting positions.  Parameters are organized into sections which tell the
system which data block to put them into.  The modules down the line look up
parameter values (which are chosen by the MCMC) in specific named blocks.

Syntax:
> mpirun -np 4 python mpi_sampler.py params.ini

"""



import numpy as np
import pymc
import pydesglue
import sys
import textfile_db
import os
import time
from mpi4py import MPI
from cvg_diagnostics import *

#Some versions of pymc mess with the default divide-by-zero behaviour in numpy
np.seterr(divide='warn', invalid='warn', over='warn', under='ignore')


PYMC_INI_SECTION = "pymc"
PIPELINE_INI_SECTION = "pipeline"
likelihood_evaluation_count = {}


def main(inifile_name):

	comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
	global likelihood_evaluation_count
	likelihood_evaluation_count[str(rank)] = 0
	

        try:
                MPI.Is_initialized()
        except:
                print "Error in MPI initialization"
                sys.exit(1)


	# Create the ini file object.
	ini = pydesglue.Inifile.from_file(inifile_name)
	#Get the parameter names from the pymc section of the ini file
	likelihood_names = ini.get("pipeline","likelihoods").split()
	quiet = ini.get(PIPELINE_INI_SECTION,"quiet","F")
	quiet = pydesglue.boolean_string(quiet)

	param_values_filename = ini.get(PIPELINE_INI_SECTION,"values")
	param_value_file = pydesglue.ParameterRangesFile.from_file(param_values_filename)
	#Based on filenames in it, load the covmat and parameter range files
	param_info = param_value_file.param_list()
	
	
	#Using the inifile, create a pipeline object
	#and make the model
	pipeline = pydesglue.LikelihoodPipeline(ini, quiet=quiet)
	try:
		model = make_model(param_info, pipeline, likelihood_names, rank,verbose = 0)
	except pymc.ZeroProbability:
		sys.stderr.write("There was an error doing the first run of the likelihood.\n")
		sys.stderr.write("This may be caused by an error in one of the physics or likelihood functions or a bad parameter in %s\n" % param_values_filename)
		sys.stderr.write("More information may be printed above\n")
		comm.Abort()
		

	#run the model!
	mcmc=  mpi_sampling(comm, model, param_info, ini)

	try:
		MPI.Is_finalized()
	except:
		print "Error in MPI finalization"
		sys.exit(1)

	return mcmc


def rescale_trace(mcmc,param_info,key):
	param_ranges = [info[2] for info in param_info]
	params = {} # dictionary for rescaling parameter ranges
	for (section, param, param_range) in param_info:
		param_code = encode_param_name(section, param)
		param_min, param_start, param_max = param_range
		if param_min==param_max:
			params[param_code] = [param_start,param_start]
		else:
                        params[param_code] = [param_min,param_max]
	trace= np.array([get_value(x) for x in mcmc.trace(key)[:]])
	value = params[key][0] + (trace)*(params[key][1]-params[key][0])
	return value
	



def output_traces(mcmc,param_info,output_file,output_header):
	if output_header:
		output_file.write('# likelihood ')
	traces = np.array([-0.5*(mcmc.trace('deviance')[:])])
	keys = mcmc.get_state()["stochastics"].keys()
	for key in keys:
		if output_header:
			output_file.write('%s  '  % key)
		value = np.array([rescale_trace(mcmc,param_info,key)])
		traces = np.concatenate((traces,value),axis=0)
	if output_header:
		output_file.write('\n')
	np.savetxt(output_file, np.c_[traces.T],fmt = '%.7e')
	output_file.flush()





def encode_param_name(section, name):
	return '%s--%s'%(section, name)

def seedgen(rank):
	s = time.time()
	return int(abs(((s*181)*((rank-83)*359))%104729))


def make_model(param_info, pipeline, likelihood_names,rank, verbose = 2):
	# create the pymc parameter objects, and put them in a dictionary by name
	params = {}
	np.random.seed(seedgen(rank))
	for (section, param, param_range) in param_info:
		param_code = encode_param_name(section, param)
		param_min, param_start, param_max = param_range
		if param_min>param_max:
			raise ValueError("In your values file you set min>max for parameter %s in section %s (%f>%f) " % (param,section,param_min, param_max) )
		elif param_start>param_max or param_start<param_min:
			raise ValueError("In your values file you set a starting value (%f) for parameter %s in section %s outside the min-max range (%f to %f)  " % (param_start, param,section,param_min, param_max) )
		elif param_min==param_max:
			params[param_code] = param_start
		else:
			print "Adding parameter:", param_code
			if (section,param) in pipeline.priors_calculator.prior_data:
				prior_type, prior_params = pipeline.priors_calculator.prior_data[(section,param)]
				if prior_type == pipeline.priors_calculator.UNIFORM:
					print "Adding additional uniform prior on ", param
					param_min = max(param_min, prior_params[0])
					param_max = min(param_max, prior_params[1])
					if not param_min<param_max:
						raise ValueError("Prior ranges set in prior file and value file are incompatible")
					start_value = np.random.uniform(low=0.0,high=1.0)
					params[param_code] = pymc.Uniform(param_code, lower = 0.0, upper = 1.0, value = start_value)
				elif prior_type == pipeline.priors_calculator.GAUSSIAN:
					print "Adding additional gaussian prior on ", param
					mu = (prior_params[0]-param_min)/(param_max-param_min)
					sigma = prior_params[1]/(param_max-param_min)
					start_value = np.random.standard_normal()
					params[param_code] = pymc.Normal(param_code, mu=mu, tau=sigma**-2,value = start_value)
				elif prior_type == pipeline.priors_calculator.EXPONENTIAL:
					print "Adding additional exponential prior on ", param
					if param_max<0:
						raise ValueError("Parameter %s maximum in values file is < 0 but exponential prior in prior file not compatible witht this"%param)
					beta = prior_params[0]/(param_max-param_min)
					start_value = np.random.standard_exponential()
					params[param_code] = pymc.Exponential(param_code, beta=1./beta, value = start_value)
				else:
					raise ValueError("Unknown prior type in pymc sampler - complain to Joe Zuntz")
			else:
				start_value = np.random.uniform(low=0.0,high=1.0)
				params[param_code] = pymc.Uniform(param_code, lower = 0.0, upper = 1.0, value = start_value)
				params[param_code]._original_lower = param_min
				params[param_code]._original_upper = param_max



	
	# set up the likelihood module, which runs the pipeline on the parameter list after extracting them
	# from the objects.  It will get the likelihood out at the end.  Maybe by name?  Need to look up.
	# Then returns this value.
	@pymc.data
	@pymc.stochastic(verbose = verbose)
	def data_likelihood(params = params, value = 0.0,rank=rank):
		global likelihood_evaluation_count
		#Read the parameters from the vector
		pipeline_sections = {}
		for (section, param, param_range) in param_info:
			pipeline_sections[section] = {}

		helps = []
		previous_section=None
		if not pipeline.quiet:
			print ("Running pipeline iteration %d  on process %s: " % (likelihood_evaluation_count[str(rank)],str(rank)))
		likelihood_evaluation_count[str(rank)] += 1
		for (section, param, param_range) in param_info:
			param_code = encode_param_name(section, param)
			param_min, param_start, param_max = param_range
			value = get_value(params[param_code])*(param_max-param_min) + param_min
			pipeline_sections[section][param] = value
			if not pipeline.quiet:
				if param_range[0]!=param_range[2]:
					print '  %s/%s=%.4g' % (section, param, value), 
		if not pipeline.quiet: print


		# Run the pipeline on these parameters
		data = pipeline.run_parameters(pipeline_sections)
		
		# If something went wrong (e.g. unphysical) we get None back, so check for this and
		# return zero probability if so:
		if data is None:
			if not pipeline.quiet: print "Likelihood  -inf"
			return -np.inf
		
		#Otherwise, use a package method to extract the list of likelihoods
		like = pipeline.extract_likelihood(data)
		if not pipeline.quiet: print "Likelihood ", like
		return like
		
	variables = {'data_likelihood':data_likelihood,'params':params}
	for param_code,param in params.items():
		if isinstance(param, pymc.Node):
			variables[param_code] = param
	return variables
		

#This is a helper function to get around an annoying
#thing in pymc where parameters have a different type the first time they
#are created than the second.
def get_value(p):
	if isinstance(p,float): return p
	if isinstance(p, int): return p
	elif isinstance(p,np.ndarray):
		if p.ndim==0: return float(p)
		else: return p[0]
	raise ValueError("Unknown type %r  (%r)" % (type(p), p))


	
def reorder_hack(old_order, new_order, cov):
	n = len(old_order)
	cov2 = np.zeros((n,n))
	for i in xrange(n):
		old_i = old_order.index(new_order[i])
		for j in xrange(n):
			old_j = old_order.index(new_order[j])
			cov2[i,j] = cov[old_i, old_j]
	return cov2
		

def normalize_covmat(covmat, param_range):
	r = np.array([p[2]-p[0] for p in param_range if p[2]!=p[0] ])
	n = covmat.shape[0]
	for i in xrange(n):
		covmat[i,:]/=r
		covmat[:,i]/=r






def run_model(model, param_info, ini,nsamples):
	covmat_filename = ini.get(PYMC_INI_SECTION, "covmat")
	covmat = np.loadtxt(covmat_filename)
	param_names =   [encode_param_name(section,name)
			for (section,name,param_range) in param_info
			if param_range[0]!=param_range[2]
			]
	for p in param_names:
		print p
	nparam = len(param_names)
	if covmat.ndim==0:
		# one parameter
		covmat = covmat.reshape((1,1))
	if covmat.ndim==1:
		print "1D covmat found.  Assuming that this was meant to be diagonal STANDARD DEVIATIONS"
		covmat = np.diag(covmat**2)
	if covmat.shape != (nparam, nparam):
		param_text = ', '.join([name for (section,name,param_range) in param_info if param_range[0]!=param_range[2]])
		info = covmat.shape + (nparam, param_text)
		raise ValueError("The covmat loaded was shape (%rx%r). We have %r non-fixed parameters so was expecting that dimension. (Parameters were: %r)" % info)


	try:
		old_output_format = ini.get(PYMC_INI_SECTION, "old_output")
	except pydesglue.ConfigParser.NoOptionError:
		old_output_format = "N"

	try:
		do_normal = ini.get(PYMC_INI_SECTION,"normal_approximation")
	except pydesglue.ConfigParser.NoOptionError:
		do_normal = "False"

	if pydesglue.boolean_string(do_normal):
		print "Fitting a normal approximation to the likelihood."
		print "Finding approx ML point and approx covariance matrix"
		db = textfile_db
		output_filename = ini.get(PYMC_INI_SECTION, "name")
		norm_approx = pymc.NormApprox(model, db=db, dbname=output_filename, eps=0.01, diff_order=3)
		norm_approx.fit()
		norm_approx.sample(nsamples)
		return norm_approx
	print "Starting MCMC"
	#Set up the MCMC.  Annoying hack here.
	mcmc=pymc.MCMC(model, db='ram',  verbose=2)
	params = [model[p] for p in param_names]
	ordering = [m.__name__ for m in mcmc.stochastics]

	#We normalized the param ranges to 0-1.  Need to scale covmat accordingly
	#We also need to reorder to match pymc's ordering
	param_ranges = [info[2] for info in param_info]
	normalize_covmat(covmat, param_ranges)
	covmat = reorder_hack(param_names, ordering, covmat)
	
	try:
		do_adaptive = ini.get(PYMC_INI_SECTION,"adaptive_mcmc")
	except pydesglue.ConfigParser.NoOptionError:
		do_adaptive = "False"
	if pydesglue.boolean_string(do_adaptive):
		mcmc.use_step_method(pymc.AdaptiveMetropolis, params, cov=covmat, interval=100, delay=100, verbose=0)
	else:
		for param in params:
			mcmc.use_step_method(pymc.Metropolis,param,verbose=0)
	return mcmc


def get_traces(mcmc,param_info,keys):
        traces = []
        for key in keys:
                traces.append((rescale_trace(mcmc,param_info,key)))
	return traces

def check_convergence(comm,mcmc,param_info,ini,totalsteps,Diag):
	keys = mcmc.get_state()["stochastics"].keys()  # stochastic variables
	traces = get_traces(mcmc,param_info,keys)
	msg = Diag.get_convergence(comm,traces,totalsteps,keys)
	return msg


def mpi_sampling(comm, model, param_info, ini):
	size = comm.Get_size()
	rank = comm.Get_rank()
	output_filename = ini.get(PYMC_INI_SECTION, "name")
        dbname = ''.join(['chain_',str(rank),'_',output_filename]) # output file name
	db = 'ram'
	samples = int(ini.get(PYMC_INI_SECTION, "samples"))
	try:
		do_gelman_rubin = ini.get(PYMC_INI_SECTION,"gelman_rubin")
	except pydesglue.ConfigParser.NoOptionError:
		do_gelman_rubin = "False"
	try:
		do_adaptive = ini.get(PYMC_INI_SECTION,"adaptive_mcmc")
	except pydesglue.ConfigParser.NoOptionError:
		do_adaptive = "False"

	if size == 1:
		do_gelman_rubin = "False"

	if pydesglue.boolean_string(do_gelman_rubin):
		if size <=1:
			print "Number of processors must be >=2 for Gelman-Rubin test"
			sys.exit(0)
		burn = float(ini.get(PYMC_INI_SECTION, "burn"))
		nsteps = int(ini.get(PYMC_INI_SECTION, "nsteps")) # check for convergence every nsteps after nburn
		Rcrit = ini.get(PYMC_INI_SECTION, "Rcrit") # check for convergence every nsteps after nburn
		nburn = int(burn*samples)
		sampling_steps = samples  - nburn

	output_file = open(dbname,'w')	
	output_header = 1

	
	if pydesglue.boolean_string(do_gelman_rubin):
		mcmc  = run_model(model,param_info,ini,sampling_steps)
		if nburn == 0:
			pass
		else:
			mcmc.sample(iter=burn,progress_bar=False,tune_throughout=False)
			output_traces(mcmc,param_info,output_file,output_header)
			output_header = 0

		totalsteps = 0
		Diag=Diagnostics(Rcrit,len(param_info)) #instance of Diagnostic class
		if  nsteps <= sampling_steps: #checking cvg every nsteps  - make sure that there are nsteps left after burn
			for i in range(int(sampling_steps/nsteps)):
				mcmc.sample(iter=nsteps,progress_bar=False,tune_throughout=False)
				totalsteps += nsteps
				msg = check_convergence(comm,mcmc,param_info,ini,totalsteps,Diag)
				output_traces(mcmc,param_info,output_file,output_header)
				output_header = 0
				if msg:
					output_file.close()
					return 0

			if sampling_steps % nsteps != 0:
				remainder = sampling_steps % nsteps
				mcmc.sample(iter=remainder,progress_bar=False,tune_throughout=False)
				totalsteps += remainder
				msg = check_convergence(comm,mcmc,param_info,ini,totalsteps,Diag)
				output_traces(mcmc,param_info,output_file,output_header)
				if msg:
					output_file.close()
					return 0
		else:
			mcmc.sample(iter=sampling_steps,progress_bar=False,tune_throughout=False)
			totalsteps += sampling_steps
			msg = check_convergence(comm,mcmc,param_info,ini,totalsteps,Diag)
			output_traces(mcmc,param_info,output_file,output_header)
			if msg:
				output_file.close()
				return 0

		if not bool(msg):
			if rank ==0 :
				print ("Output %s samples and chains may not converged"% str(totalsteps))
			output_file.close()
		return 0
	else:
		mcmc  = run_model(model,param_info,ini,samples)
		mcmc.sample(iter=samples,progress_bar=False,tune_throughout=False)
		output_traces(mcmc,param_info,output_file,output_header)
		output_file.close()
		return 0
	


	
def usage():
	sys.stderr.write(__doc__)
	
if __name__=="__main__":
	try:
		inifile = sys.argv[1]
	except IndexError:
		usage()
		sys.exit(1)
	main(inifile)
	# test():
