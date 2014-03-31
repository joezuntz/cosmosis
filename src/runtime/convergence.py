import numpy as np
import pymc
import sys
import os
from mpi4py import  MPI



class Diagnostics(object):
	def __init__(self,Rcrit,nparams):
		self.rcrit = Rcrit
		self.totalsteps = 0
		self.traces = []
		self.means = [0]*nparams
		self.variances = [0]*nparams
		self.keys = []


	def write_diagnostic_file(self,all_diagnostics,flag):
		output_file = open('gelman_rubin_txt','w')
		if not bool(flag): output_file.write('Warning:  chains may not have converged as R statistic is greater then %s \n'%self.rcrit)
		output_file.write('number of steps in chain =  %s \t R = %s \n'% (self.totalsteps,self.rcrit))
		output_file.write('Parameter \t between chain \t within chain \t R estimate \n')
		i=0
		for key in self.keys:
			output_file.write('%s \t %.4f \t %.4f \t%.4f \n '% (key,all_diagnostics[i][0], all_diagnostics[i][1],all_diagnostics[i][2]))
			i += 1


	def get_rstat(self,varchains, meanchains,size):
		"""Brooks and Gelman (1998), Gelman and Rubin (1992)"""
		B_over_n = ( np.var(meanchains,ddof=1))
		B = B_over_n*self.totalsteps
		W = 1.0/size*(np.sum(varchains))
		mpv = (self.totalsteps - 1)/self.totalsteps * W + B_over_n  #marginal posterior variance of parameter
		V = mpv + B_over_n/size
		if W ==0:
			return [B,W,np.inf] # zero variance in chain
		else:
			Rhat = V/W
			diagnostics = [B,W,Rhat]
			return diagnostics


	def gather_stats(self,comm):
		size = comm.Get_size()
		rank = comm.Get_rank()
		try:
			varchains = comm.gather(self.variances, root=0)  #gather list of variances for each parameter to root
		except:
			print "MPI gather error"
			comm.Abort()
		try:
			meanchains = comm.gather(self.means, root=0)  #gather list of means for each parameter to root
		except:
			print "MPI gather error"
			comm.Abort()
		varchains = np.array(varchains)
		single_parameter_variances = varchains.T
		meanchains = np.array(meanchains)
		single_parameter_means = meanchains.T
		if rank == 0:
			all_diagnostics=[]
			flag = 1
			for i in range(len(single_parameter_variances)):
				#check for convergence of all parameters
				diagnostics =  self.get_rstat(single_parameter_variances[i],single_parameter_means[i],size)
				all_diagnostics.append(diagnostics)
				if diagnostics[2] <= self.rcrit:
					pass
				else:
					flag =0  # continue chain
			self.write_diagnostic_file(all_diagnostics,flag)
		else:
			flag = 0
		# broadcast the stop or continue message
		try:
			msg = comm.bcast(flag,root=0)
			return msg
		except:
			print "MPI broadcast error"
			comm.Abort()
	


	def update_moments(self,trace,mean,M2):
		num=self.totalsteps-len(trace)
		for x in trace:
			num = num + 1
			delta  = x - mean
			mean  = mean + delta/num
			M2 = M2 + delta*(x-mean)
			if num >1:
				s = M2/(num-1)
			else:
				s=0
		return mean,s


	def get_moments(self):
		mu_new = []
		sigma_new = []
		for i in range(len(self.traces)):
			m,s = self.update_moments(self.traces[i],self.means[i],self.variances[i])
			mu_new.append(m)
			sigma_new.append(s)
		return mu_new , sigma_new


	def get_convergence(self,comm,traces,totalsteps,keys):
		"""mpi_sampler checks for convergence every len(traces), totalsteps are total in chain so far"""
		self.traces=traces
		self.totalsteps=totalsteps
		self.keys=keys
		self.means,self.variances = self.get_moments()
		msg = self.gather_stats(comm)
		return msg

	@staticmethod
	def gelman_rubin(traces):
		"""IO module reads in finished  chains for one parameter"""
		x=traces
		m,n= np.shape(traces)
		B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)
		W = np.sum([(x[i] - xbar) ** 2 for i,xbar in enumerate(np.mean(x,1))]) / (m * (n - 1))
		s2 = W * (n - 1) / n + B_over_n
		V = s2 + B_over_n / m
		if W ==0:
			return [B_over_n*n,W,np.inf] # zero variance in chain
		else:
			R = V/W
			return [B_over_n*n,W,R]

	@staticmethod
	def finished_chain_diag(trace):
		"""convergence diagnostics on one trace"""
		z_scores = pymc.geweke(trace, first=.1, last=.5, intervals=20)
		RL_output = pymc.raftery_lewis(trace, 0.025, 0.05, s=.95, epsilon=.001, verbose=1)
		return z_scores

		

