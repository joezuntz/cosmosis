#!/usr/bin/env python
import numpy as np
import pydesglue
import sys
import emcee
from mpi4py import MPI
from emcee.utils import MPIPool
import os

EMCEE_INI_SECTION = "emcee"


def master_task(comm, pool, resume=None):
	pipeline = pydesglue.LikelihoodPipeline(inifile, id=comm.Get_rank())
	trues = ['T','True','true','TRUE','Y','y','yes','Yes','YES']
	debug = pipeline.get_option(EMCEE_INI_SECTION, "debug", default='no') in trues

	param_names = ["%s--%s"%(p[0],p[1]) for p in pipeline.varied_params]
	nwalkers = int(pipeline.get_option(EMCEE_INI_SECTION, "walkers"))
	nsample = int(pipeline.get_option(EMCEE_INI_SECTION, "samples"))
	output_filename = pipeline.get_option(EMCEE_INI_SECTION, "outfile")
	ndim = len(pipeline.varied_params)
	file_mode = "w"

	if resume is not None:
		if resume=="--resume":
			if os.path.exists(output_filename):
				file_mode = "a"
				p0 = [row for row in np.loadtxt(output_filename)[-nwalkers:,1:ndim+1]]
			else:
				p0 = [pipeline.randomized_start() for i in xrange(nwalkers)]
		else:
			p0 = [row for row in np.loadtxt(resume)[-nwalkers:,1:ndim+1]]
			if len(p0)<nwalkers:
				for i in xrange(len(p0),nwalkers):
					p0.append(pipeline.randomized_start())
	else:
		p0 = [pipeline.randomized_start() for i in xrange(nwalkers)]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, worker_task, pool=pool)

	extra_info_header = ['%s--%s'%param for param in pipeline.extra_saves]

	f = open(output_filename, file_mode)
	if file_mode=="w":
		f.write('# LIKE         \t')
		f.write('\t'.join(["%-20s"%p for p in param_names]))
		f.write('\t')
		f.write('\t'.join(["%-20s"%p for p in extra_info_header]))
		f.write('\n')
	for pos, prob, rstate, extra_info in sampler.sample(p0, iterations = nsample):
		for row,extra in zip(pos,extra_info):
			row_string="\t".join(["%-20s"%str(value) for value in row])
			extra_string="\t".join(["%-20s"%str(extra[param]) for param in extra_info_header])
			like=extra["LIKE"]
			f.write('%le\t%s\t%s\n'%(like,row_string,extra_string))
		f.flush()
	f.close()

my_pipeline = None
def worker_task(task):
	return my_pipeline.likelihood(task)


def main(inifile_name, resume=None):
	global my_pipeline
	comm = MPI.COMM_WORLD
	pool = MPIPool(comm)
	
	if pool.is_master():
		master_task(comm, pool, resume)
	else:
		my_pipeline = pydesglue.LikelihoodPipeline(inifile, id=comm.Get_rank() )
		pool.wait()
	pool.close()




USAGE_MESSAGE = """
This sampler uses the EMCEE package to sample over the parameters.
The pipeline is specified in the parameter file and the parameters in 
a second file reference in that one.

Syntax:  mpirun -np <number-of-processors> python emcee_sampler.py params.ini
"""
	
	
if __name__=="__main__":
	try:
		inifile = sys.argv[1]
	except IndexError:
		sys.stderr.write(USAGE_MESSAGE)
		sys.exit(1)
	resume=None
	if len(sys.argv)>2:
		resume=sys.argv[2]
	main(inifile, resume)
