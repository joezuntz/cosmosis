from multi_text_output import MultiTextOutput
import string
import numpy as np
import os
import shutil
from mpi4py import MPI
import pymc

try:
	import analytics
except:
	analytics = None

def populate_table(out, nparam, ns):
	out.metadata('NP',nparam)
	out.metadata('NS',ns)
	out.metadata('TIME','1:30pm')

	for i in xrange(nparam):
		p = string.ascii_uppercase[i]
		out.add_column(p, float, 'The parameter called %s'%p)

	for i in xrange(ns):
		x = np.arange(nparam, dtype=int)+np.random.randint(1,high=10)
		out.parameters(x)
	out.final("FINISH",True)
	out.close()

def test_text():
	comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
	dirname='temp_output_test'
	msg = 1 
	if rank==0:
		if os.path.exists(dirname):
			shutil.rmtree(dirname)
			os.mkdir(dirname)
		else:
			os.mkdir(dirname)
	msg = comm.bcast(msg,root=0)

	ini = {'filename':dirname, 'format':'multitext','nchain':rank}
	out = MultiTextOutput.from_options(ini)
	nparam = 8
	ns = 50 
	populate_table(out, nparam, ns)

#may need to also savestate if PyMC adaptive method used



#read chains after sampling cvg tests or plotting should  use these read functions?
	if rank ==0:
		chains = MultiTextOutput.get_chains(dirname)
		traces = np.ndarray(shape=(len(chains),ns),dtype=float)
		for i in np.arange(len(chains)):
			traces[i] = MultiTextOutput.load_txt_tables(chains[i])[0] # e.g. collect all traces for first parameter

		if cvg_diagnostics:
			B,W,R = cvg_diagnostics.Diagnostics.gelman_rubin(traces)
			print("Gelman Rubin R = %f \n"% R)
			z = cvg_diagnostics.Diagnostics.finished_chain_diag(traces[0])
			print "z_scores",z

	


if __name__=="__main__":
	test_text()
