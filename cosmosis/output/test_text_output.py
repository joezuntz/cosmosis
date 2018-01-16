from builtins import range
from .text_output import TextColumnOutput
import string
import numpy as np
import os
try:
	import astropy.table
except:
	astropy = None

def populate_table(out, nparam, ns):
	out.metadata('NP',nparam)
	out.metadata('NS',ns)
	out.metadata('TIME','1:30pm')

	for i in range(nparam):
		p = string.ascii_uppercase[i]
		out.add_column(p, float, 'The parameter called %s'%p)

	for i in range(ns):
		x = np.arange(nparam, dtype=int)+i
		out.parameters(x)
	out.final("FINISH",True)
	out.close()

def test_text():
	filename='temp_output_test.txt'
	ini = {'filename':filename, 'format':'text'}
	out = TextColumnOutput.from_options(ini)
	nparam = 8
	ns = 20 
	populate_table(out, nparam, ns)

	#We should be able to load this table with loadtxt
	if astropy:
		t = astropy.table.Table.read(filename, format='ascii.commented_header')
		A = t['A']
		B = t['B']
	else:
		t = np.loadtxt(filename,dtype=int).T
		A = t[0]
		B = t[1]

		assert A == np.arange(ns, dtype=int)
		assert B == np.arange(ns, dtype=int)+1


	#or with our own method
	names, cols, meta, final = TextColumnOutput.load(filename)
	assert names == [string.ascii_uppercase[i] for i in range(nparam)]
	assert len(cols)==nparam
	assert len(cols[0])==ns
	assert meta['NP']==nparam
	assert final['FINISH'] is True

	os.remove(filename)

