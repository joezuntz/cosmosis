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
		cube_p[ndim+i] = extra[i]
	return like

def dumper(nsample, nlive, nparam, live, post, paramConstr, max_log_like, logz, ins_logz, log_z_err, context):
	print "Saving %d samples" % nsample
	sampler.output_params(nsample, live, post, logz, ins_logz, log_z_err)


class MultinestSampler(Sampler):
	sampler_outputs = [("like", float), ("weight", float)]

	def config(self):
		global libnest3
		if libnest3 is None:
			try:
				libnest3 = ct.cdll.LoadLibrary(libname)
			except Exception as error:
				sys.stderr.write("Multinest could not be loaded correctly.\n")
				sys.stderr.write("This may mean an MPI compiler was not found to compile it,\n")
				sys.stderr.write("or that some other error occurred.  More info below.\n")
				sys.stderr.write(str(error)+'\n')
				sys.exit(1)
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

		#Required options
		self.max_iterations = self.read_ini("max_iterations", int)
		self.live_points    = self.read_ini("live_points", int)

		#Output and feedback options
		self.feedback               = self.read_ini("feedback", bool, True)
		self.resume                 = self.read_ini("resume", bool, False)
		self.multinest_outfile_root = self.read_ini("multinest_outfile_root", str, "")
		self.update_interval        = self.read_ini("update_interval", int, 200)

		#General run options
		self.random_seed = self.read_ini("random_seed", int, -1)
		self.importance  = self.read_ini("ins", bool, True)
		self.efficiency  = self.read_ini("efficiency", float, 1.0)
		self.tolerance   = self.read_ini("tolerance", float, 0.1)
		self.log_zero    = self.read_ini("log_zero", float, -1e6)

		#Multi-modal options
		self.mode_separation    = self.read_ini("mode_separation", bool, False)
		self.const_efficiency   = self.read_ini("constant_efficiency", bool, False)
		self.max_modes          = self.read_ini("max_modes", int, default=100)
		self.cluster_dimensions = self.read_ini("cluster_dimensions", int, default=-1)
		self.mode_ztolerance    = self.read_ini("mode_ztolerance", float, default=0.5)

	@classmethod
 	def needs_output(cls):
		try:
			import mpi4py.MPI
		except Exception as error:
				sys.stderr.write("mpi4py could not be imported or loaded.\n")
				sys.stderr.write("Our multinest interface needs it.\n")
				sys.stderr.write("You could try: pip install mpi4py\n")
				sys.stderr.write("More info below:")
				sys.stderr.write(str(error)+'\n')
				sys.exit(1)
		rank = mpi4py.MPI.COMM_WORLD.Get_rank()
		return (rank==0)


	def execute(self):

		cluster_dimensions = self.ndim if self.cluster_dimensions==-1 else self.cluster_dimensions
		periodic_boundaries = (ct.c_int*self.ndim)()
		context=None
		wrapped_likelihood = loglike_type(likelihood)
		wrapped_output_logger = dumper_type(dumper)
		try:
			import mpi4py.MPI
		except Exception as error:
				sys.stderr.write("mpi4py could not be imported or loaded.\n")
				sys.stderr.write("Our multinest interface needs it.\n")
				sys.stderr.write("You could try: pip install mpi4py\n")
				sys.stderr.write("More info below:")
				sys.stderr.write(str(error)+'\n')
				sys.exit(1)

		init_mpi=False
		self.log_z = 0.0
		self.log_z_err = 0.0

		self._run(self.importance, self.mode_separation, self.const_efficiency, self.live_points, self.tolerance, self.efficiency,
			self.ndim, self.npar, cluster_dimensions, self.max_modes, self.update_interval,
			self.mode_ztolerance, self.multinest_outfile_root, self.random_seed, periodic_boundaries, self.feedback,
			self.resume, self.multinest_outfile_root!="", init_mpi, self.log_zero, self.max_iterations, wrapped_likelihood, 
			wrapped_output_logger, context)

		if self.output:
			self.output.final("log_z", self.log_z)
			self.output.final("log_z_error", self.log_z_err)
		self.converged = True


	def output_params(self, n, live, posterior, log_z, ins_log_z, log_z_err):
		self.log_z = ins_log_z if self.importance else log_z
		self.log_z_err = log_z_err
		data = np.array([posterior[i] for i in xrange(n*(self.npar+2))]).reshape((self.npar+2, n))
		for row in data.T:
			params = row[:self.ndim]
			extra_vals = row[self.ndim:self.npar]
			like = row[self.npar]
			importance = row[self.npar+1]
			self.output.parameters(params, extra_vals, like, importance)
		self.output.final("nsample", n)

	def is_converged(self):
		return self.converged

