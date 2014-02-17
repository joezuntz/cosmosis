#!/usr/bin/env python
import numpy as np
import pydesglue
import sys
import scipy.optimize
import scipy.misc
import pylab

ROOT2=np.sqrt(2)
covmat_estimate = False #not working yet

def line_function(x, pipeline, p0, i, j):
	p = p0.copy()
	if i==j:
		p[i] += x
	else:
		p[i] += x/ROOT2
		p[j] += x/ROOT2
	q = pipeline.denormalize_vector(p)
	# print "XXX", p
	# return  -(0.5 * (p-0.5)**2 / 0.1**2).sum()

	return pipeline.posterior(q)[0]


def estimate_second_derivative_about_zero(function, args):
	#find the scale at which delta-L ~ 1
	# print "getting L0"
	L0 = function(0.0, *args)
	# print "L0 = ", L0
	scale=None
	for logx in [-5,-4,-3,-2,-1]:
		x = 10.0**logx
		L = function(x, *args)
		dL = L0-L
		print 'scaling: ', x, dL
		if np.log10(dL) > -1 and np.log10(dL)<1:
			scale=x/2.0
			break
	else:
		raise ValueError("Fisher code failure")
	X = np.arange(-3,4)*scale
	Y = [function(x,*args) for x in X]
	X = np.array(X)
	Y = np.array(Y)
	w = np.where(np.isfinite(Y))
	X = X[w]
	Y = Y[w]
	print "X:", X
	print "Y:", Y
	p = np.polyfit(X,Y,2)
	p2,p1,p0 = p
	# pylab.figure()
	# pylab.plot(X,Y,'b.')
	# X_fit = np.linspace(X[0],X[-1],100)
	# Y_fit = np.polyval(p, X_fit)
	# print 'X_fit: ', X_fit
	# pylab.plot(X_fit,Y_fit,'r-')
	# pylab.title('%d  %d' % (args[2],args[3]))
	#p1 should be near zero.  p0 
	print "These should close:", p0, L0
	return p2




def estimate_covariance_matrix(p_in, pipeline):
	n = len(p_in)
	p = pipeline.normalize_vector(p_in)
	F = np.zeros((n,n))
	for i in xrange(n):
		for j in xrange(i+1):
			print "COMPUTING FISHER", i, j
			# F[i,j] = F[j,i] = scipy.misc.derivative(line_function, 0.0, dx=1e-6, n=2, order=5, args=(pipeline, p, i, j))
			F[i,j] = F[j,i] = estimate_second_derivative_about_zero(line_function, args=(pipeline, p, i, j))
	for i in xrange(n):
		F[i,i]= -2*F[i,i]
	for i in xrange(n):
		for j in xrange(n):
			if i!=j:
				F[i,j] = -2*F[i,j]-0.5*(F[i,i]+F[j,j])
	print "Eigenvals:", np.linalg.eigvals(F)
	# F = np.linalg.inv(F)
	F-=2*np.linalg.eigvals(F).min()*np.eye(F.shape[0])
	for i in xrange(n):
		range_i = pipeline.varied_params[i][2]
		delta_i = range_i[2] - range_i[0]
		for j in xrange(n):
			range_j = pipeline.varied_params[j][2]
			delta_j = range_j[2] - range_j[0]
			print 'F, i, j, scaling = ', F[i,j], i, j, delta_i * delta_j
			F[i,j] *= delta_i * delta_j
	covmat = np.linalg.inv(F)
	return covmat

def my_like(p_in, pipeline):
	p = pipeline.denormalize_vector(p_in)
	like = -pipeline.likelihood(p)[0]
	# print p_in
	# like = (0.5 * (p_in-0.5)**2 / 0.1**2).sum()
	# print ' '.join(str(x) for x in p), '  like = ', like
	return like

def main(args):
	if len(args)==1:
		inifile_name = args[0]
		output_file = sys.stdout
		fisher=None
	elif len(args)==2:
		inifile_name = args[0]
		output_file = args[1]
		fisher=None
	elif len(args)==3:
		inifile_name = args[0]
		output_file = args[1]
		fisher=args[2]	
	else:
		raise ValueError("Syntax: python maxlike.py params.ini [output_file] [fisher_file]")
	print inifile_name


	pipeline = pydesglue.LikelihoodPipeline(inifile_name)
	start = np.array([p[2][1] for p in pipeline.varied_params])
	print "START:"
	print start
	norm_start = pipeline.normalize_vector(start)
	opt_norm = scipy.optimize.fmin(my_like, norm_start, args=(pipeline,), xtol=0.001)
	print "END\n"
	opt = pipeline.denormalize_vector(opt_norm)
	print
	pipeline.write_values_file(opt,output_file)
	if covmat_estimate:
		print "Sorry, covmat estimator not working yet"
		return opt, None
		C = estimate_covariance_matrix(opt, pipeline)
		np.savetxt(covmat_estimate, C)		
		return opt, C
	else:
		return opt

if __name__=="__main__":
	main(sys.argv[1:])
