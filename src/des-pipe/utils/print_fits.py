#!/usr/bin/env python
import pyfits
import sys
import re
import warnings

EXCLUDED_PARAMETERS = ["XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", 
"PCOUNT", "GCOUNT", "TFIELDS", "EXTNAME", "TTYPE", "TFORM", "TUNIT","SIMPLE", "EXTEND"]


def print_parameters(header):
	pairs = []
	for key in header:
		ex = False
		for exclude in EXCLUDED_PARAMETERS:
			if key.startswith(exclude): ex = True
		if ex: continue
		try:
			value = header[key]
		except:
			warnings.warn("Your version of PYFITS is a bit old and has trouble with long FITS param names like %s"%key)
			continue
		if key.startswith("HIERARCH"): key = key[8:]
		pairs.append((key,value))

	if pairs:
		print "Params:"
		for (key,value) in pairs:
			print '    %10s = %10r' % (key, value)

def print_data(data):
	output = []
	n=0
	for col in data.columns:
		if col.name =="_DUMMY":
			continue
		col_data = data.field(col.name)
		if not output:
			n = col_data.shape
		if len(n)>1:
			out = "<%%d-dimensional array>" % (len(n))
		else:
			if n[0]>=5:
				out = "    %8s =  %10g,  %10g,  %10g,  %10g,  %10g,  ...,  %10g" % (col.name, col_data[0], col_data[1], col_data[2], col_data[3], col_data[4],col_data[-1])
			else:
				out = ("    %8s = " + ',  '.join(["%10g" for i in xrange(n[0])])) % tuple([col.name] + [col_data[i] for i in xrange(n[0])])
		output.append(out)
	if output:
		print "Data [%s]:" % n
		for out in output:
			print out

def print_fits_file(filename):
	print '###################'
	print filename
	print '###################'
	for ext in pyfits.open(filename)[1:]:
		print '-'*10
		print ext.name
		print '-'*10
		print_parameters(ext.header)
		print_data(ext.data)
		print
	print
	print

if __name__=="__main__":
	for filename in sys.argv[1:]:
		print_fits_file(filename)
