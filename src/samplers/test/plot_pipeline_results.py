"""
This code make some plots from the result of the default run_pipeline.py
script.  It makes CMB plots and n(z) plots by redshift bin, generated from CAMB
and Donnacha Kirk's n(z) module respectively.

Its input is the FITS file that was the output of run_pipeline.py
If you change the pipeline modules run then this script will not 
work.

The script needs matplotlib, which you may not have as it is not needed by the
rest of the pipeline.

Synax is:
> python plot_pipeline_results.py fits_filename

"""
import pyfits
import matplotlib
matplotlib.use("Agg")
import pylab
import sys

def usage():
	sys.stderr.write(__doc__)

def plot_nz(ext):
	data = ext.data
	nbin=5
	z = data['Z']
	for i in xrange(1,nbin+1):
		column = 'BIN_%d' % i
		nz = data[column]
		pylab.plot(z,nz,label=column)
	pylab.xlabel("Redshift z")
	pylab.ylabel("Number density n(z)")
	pylab.legend()
	pylab.savefig("number_density.png")
	pylab.close()

def plot_da(ext):
	data = ext.data
	z = data['Z']
	da = data['D_A']
	pylab.plot(z,da)
	pylab.xlabel("Redshift z")
	pylab.ylabel("Angular Diameter Distance DA(z)")
	pylab.savefig("angular_diameter_distance.png")
	pylab.close()

def plot_pk(ext):
	data = ext.data
	params = ext.header
	nk = params['NK']
	nz = params['NZ']
	shape = (nk,nz)
	Z = data['Z'].reshape(shape)[0,:]
	K = data['K_H'].reshape(shape)[:,0]
	P = data['P_K'].reshape(shape)
	for i in xrange(0,nz,nz//5):
		z = Z[i]
		label = 'z = %.2f' % z
		pylab.loglog(K,P[:,i], label=label)
	pylab.legend(loc='lower left')
	pylab.xlabel("k/h in Mpc")
	pylab.ylabel("Linear Matter power spectrum")
	pylab.savefig("linear_matter_power.png")
	pylab.close()


def plot_cmb(ext):
	data = ext.data
	ell = data['ELL']
	tt = data['TT']
	pylab.plot(ell, tt)
	pylab.xlabel("Ell")
	pylab.ylabel("CMB Temperature Power Spectrum")
	pylab.savefig("cmb_temperature.png")
	pylab.close()

def main(filename):
	F = pyfits.open(filename)
	plot_nz(F['NZ_WL'])
	plot_da(F['DISTANCES'])
	plot_pk(F['PK_LIN'])
	plot_cmb(F['CMB_CL'])


if __name__=="__main__":
	try:
		filename = sys.argv[1]
	except IndexError:
		usage()
		sys.exit(1)
	main(sys.argv[1])