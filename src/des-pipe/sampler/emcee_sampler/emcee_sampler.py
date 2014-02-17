#!/usr/bin/env python
import numpy as np
import pydesglue
import sys
import emcee
import os

EMCEE_INI_SECTION = "emcee"
my_pipeline = None



				

def log_probability_function(p):
	return my_pipeline.posterior(p)



def main(inifile_name, resume=None):
	global my_pipeline
	my_pipeline = pydesglue.LikelihoodPipeline(inifile_name)
	param_names = ["%s--%s"%(p[0],p[1]) for p in my_pipeline.varied_params]
	ndim = len(my_pipeline.varied_params)
	print "Varying %d params" % ndim
	nwalkers = int(my_pipeline.get_option(EMCEE_INI_SECTION, "walkers"))
	nsample = int(my_pipeline.get_option(EMCEE_INI_SECTION, "samples"))
	threads = int(my_pipeline.get_option(EMCEE_INI_SECTION, "threads", default=1))
	output_filename = my_pipeline.get_option(EMCEE_INI_SECTION, "outfile")
	file_mode = "w"
	if resume is not None:
		if resume=="--resume":
			if os.path.exists(output_filename):
				file_mode = "a"
				p0 = [row for row in np.loadtxt(output_filename)[-nwalkers:,1:ndim+1]]
			else:
				p0 = [my_pipeline.randomized_start() for i in xrange(nwalkers)]
		else:
			p0 = [row for row in np.loadtxt(resume)[-nwalkers:,1:ndim+1]]
			if len(p0)<nwalkers:
				for i in xrange(len(p0),nwalkers):
					p0.append(my_pipeline.randomized_start())
	else:
		p0 = [my_pipeline.randomized_start() for i in xrange(nwalkers)]

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_function, threads=threads)

	extra_info_header = ['%s--%s'%param for param in my_pipeline.extra_saves]
#	extra_info_header.append("LIKE")


	f = open(output_filename, file_mode)
	if file_mode=="w":
		f.write('# likelihood   \t')
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



	

USAGE_MESSAGE = """
This sampler uses the EMCEE package to sample over the parameters.
The pipeline is specified in the parameter file and the parameters in 
a second file reference in that one.

Syntax:  python emcee_sampler.py params.ini
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
