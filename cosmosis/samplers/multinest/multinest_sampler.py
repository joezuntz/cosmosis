#coding: utf-8
from .. import Sampler
import ctypes as ct
import os
import cosmosis
import numpy as np
import sys

dirname = os.path.split(__file__)[0]
libname = os.path.join(dirname, "multinest_src", "libnest3.so")
libnest3 = None

loglike_type = ct.CFUNCTYPE(ct.c_double, 
	ct.POINTER(ct.c_double),  #cube
	ct.c_int,   #ndim
	ct.c_int,   #npars
	ct.c_voidp, #context
)


dumper_type = ct.CFUNCTYPE(None, #void
	ct.c_int,  #nsamples
	ct.c_int,  #nlive
	ct.c_int,  #npars
	ct.POINTER(ct.c_double),   #physlive
	ct.POINTER(ct.c_double),   #posterior
	ct.POINTER(ct.c_double),   #paramConstr
	ct.c_double,   #maxLogLike
	ct.c_double,   #logZ
	ct.c_double,   #INSLogZ
	ct.c_double,   #logZerr
	ct.c_voidp,   #Context
)


multinest_args = [
	ct.c_bool,  #nest_IS   nested importance sampling 
	ct.c_bool,  #nest_mmodal   mode separation
	ct.c_bool,  #nest_ceff   constant efficiency mode
	ct.c_int,   #nest_nlive
	ct.c_double, #nest_tol
	ct.c_double, #nest_ef
	ct.c_int, #nest_ndims,
	ct.c_int, #nest_totPar,
	ct.c_int, #nest_nCdims,
	ct.c_int, #maxClst,
	ct.c_int, #nest_updInt,
	ct.c_double, #nest_Ztol,
	ct.c_char_p, #nest_root,
	ct.c_int, #seed,
	ct.POINTER(ct.c_int), #nest_pWrap,
	ct.c_bool, #nest_fb,
	ct.c_bool, #nest_resume,
	ct.c_bool, #nest_outfile,
	ct.c_bool, #initMPI,
	ct.c_double, #nest_logZero,
	ct.c_int, #nest_maxIter,
	loglike_type, #loglike,
	dumper_type, #dumper,
	ct.c_voidp, #context
]

MULTINEST_SECTION='multinest'
pipeline = None
sampler = None

def likelihood(cube_p, ndim, nparam, context_p):
	nextra = nparam-ndim
	#pull out values from cube
	cube_vector = np.array([cube_p[i] for i in xrange(ndim)])
	vector = pipeline.denormalize_vector(cube_vector)
	extra_names = ['%s--%s'%p for p in pipeline.extra_saves]
	try:
		like, extra = pipeline.likelihood(vector)
	except KeyboardInterrupt:
		raise sys.exit(1)
	

	for i in xrange(ndim):
		cube_p[i] = vector[i]
	
	for i in xrange(nextra):
		cube_p[ndim+i] = extra[pipeline.extra_saves[i]]
	return like

def dumper(nsample, nlive, nparam, live, post, paramConstr, max_log_like, logz, ins_logz, log_z_err, context):
	sampler.output_params(nsample, post, logz, ins_logz, log_z_err)


class MultinestSampler(Sampler):
	def config(self):
		global libnest3
		if libnest3 is None:
			libnest3 = ct.cdll.LoadLibrary(libname)
			self._run = libnest3.run
			self._run.restype=None
			self._run.argtypes = multinest_args
		self.converged=False
		global pipeline
		global sampler
		pipeline=self.pipeline
		sampler=self  #this is bad and temporary!
		self.ndim = len(self.pipeline.varied_params)
		self.npar = self.ndim + len(self.pipeline.extra_saves)

		self.output.add_column("importance", float)

		#Required options
		self.max_iterations = self.ini.getint(MULTINEST_SECTION, "max_iterations")
		self.live_points    = self.ini.getint(MULTINEST_SECTION, "live_points")

		#Output and feedback options
		self.feedback               = self.ini.getboolean(MULTINEST_SECTION, "feedback",               default=True)
		self.resume                 = self.ini.getboolean(MULTINEST_SECTION, "resume",                 default=False)
		self.multinest_outfile_root = self.ini.get(MULTINEST_SECTION,        "multinest_outfile_root", default="")
		self.update_interval        = self.ini.getint(MULTINEST_SECTION,     "update_interval",        default=200)

		#General run options
		self.random_seed = self.ini.getint(MULTINEST_SECTION,     "random_seed", default=-1)
		self.importance  = self.ini.getboolean(MULTINEST_SECTION, "ins",         default=True)
		self.efficiency  = self.ini.getfloat(MULTINEST_SECTION,   "efficiency",  default=1.0)
		self.tolerance   = self.ini.getfloat(MULTINEST_SECTION,   "tolerance",   default=0.1)
		self.log_zero    = self.ini.getfloat(MULTINEST_SECTION,   "log_zero",    default=-1e6)

		#Multi-modal options
		self.mode_separation    = self.ini.getboolean(MULTINEST_SECTION, "mode_separation",     default=False)
		self.const_efficiency   = self.ini.getboolean(MULTINEST_SECTION, "constant_efficiency", default=False)
		self.max_modes          = self.ini.getint(MULTINEST_SECTION,     "max_modes",           default=100)
		self.cluster_dimensions = self.ini.getint(MULTINEST_SECTION,     "cluster_dimensions",  default=-1)
		self.mode_ztolerance    = self.ini.getfloat(MULTINEST_SECTION,   "mode_ztolerance",     default=0.5)

 	

	def execute(self):

		cluster_dimensions = self.ndim if self.cluster_dimensions==-1 else self.cluster_dimensions
		periodic_boundaries = (ct.c_int*self.ndim)()
		context=None
		wrapped_likelihood = loglike_type(likelihood)
		wrapped_output_logger = dumper_type(dumper)
		import mpi4py.MPI
		init_mpi=False
		self.log_z = 0.0
		self.log_z_err = 0.0

		self._run(self.importance, self.mode_separation, self.const_efficiency, self.live_points, self.tolerance, self.efficiency,
			self.ndim, self.npar, cluster_dimensions, self.max_modes, self.update_interval,
			self.mode_ztolerance, self.multinest_outfile_root, self.random_seed, periodic_boundaries, self.feedback,
			self.resume, self.multinest_outfile_root!="", init_mpi, self.log_zero, self.max_iterations, wrapped_likelihood, 
			wrapped_output_logger, context)

		self.output.final("log_z", self.log_z)
		self.output.final("log_z_error", self.log_z_err)
		self.converged = True


	def output_params(self, n, posterior, log_z, ins_log_z, log_z_err):
		self.log_z = ins_log_z if self.importance else log_z
		self.log_z_err = log_z_err
		data = np.array([posterior[i] for i in xrange(n*(self.npar+2))]).reshape((self.npar+2, n))
		for row in data.T:
			params = row[:self.ndim]
			extra_vals = row[self.ndim:self.npar]
			like = row[self.npar]
			importance = row[self.npar+1]
			extra_dict = {'%s--%s'%p:v for (p,v) in zip(self.pipeline.extra_saves,extra_vals)}
			extra_dict["LIKE"] = like
			extra_dict["importance"] = importance
			self.output.parameters(params, extra_dict)

	def is_converged(self):
		return self.converged

