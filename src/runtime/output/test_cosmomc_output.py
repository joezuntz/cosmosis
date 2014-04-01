from cosmomc_output import CosmoMCOutput
import string
import numpy as np
import os
import shutil
from mpi4py import MPI
import pymc
try:
	import astropy.table
except:
	astropy = None
try:
	import cvg_diagnostics
except:
	cvg_diagnostics = None

def populate_table(out, nparam, ns):
	out.add_column('weight', float, 'The parameter called %s'%'weight')
	out.add_column('like', float, 'The parameter called %s'%'like')
	for i in xrange(nparam):
		p = string.ascii_uppercase[i]
		out.add_column(p, float, 'The parameter called %s'%p)

	for i in xrange(ns):
		like = np.ones(nparam+2)
		x = np.arange(nparam, dtype=int)+np.random.randint(1,high=10)
		like[2:] = x
		out.parameters(like)
	out.close()

def test_text():
	comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
	dirname='test_dir'
	filename ='test'
	msg = 1 
	if rank==0:
		if os.path.exists(dirname):
			shutil.rmtree(dirname)
			os.mkdir(dirname)
		else:
			os.mkdir(dirname)
	msg = comm.bcast(msg,root=0)

	ini = {'dirname':dirname,'filename':filename, 'format':'multitext','nchain':rank,'mpi':True}
	out = CosmoMCOutput.from_ini(ini)
	nparam = 10
	ns = 10 
	populate_table(out, nparam, ns)

#may need to also savestate if PyMC adaptive method used


if __name__=="__main__":
	test_text()
