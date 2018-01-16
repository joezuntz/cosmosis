from __future__ import print_function
from builtins import zip
from builtins import str
from .. import ParallelSampler
try:
    import abcpmc
    from abcpmc import mpi_util
except ImportError:
    raise ValueError("To use ABC PMC you need to install it with pip install abcpmc")
import numpy as np
from ...runtime import prior

def abc_model(p):
    data = abc_pipeline.run_parameters(p)
    if data is None:
        return None
    try:
        model = data[names.data_vector, 'simulation']
    except BlockError:
        raise ValueError("The module in the AB pipeline should save a model with the same dimensions as the data") 
    return model


class ABCSampler(ParallelSampler):
    parallel_output = False #True for 
    sampler_outputs = [("weight", float),("distance",float)]

    def config(self):
        global abc_pipeline
        abc_pipeline = self.pipeline


        if self.is_master():
            if self.pool is not None:
                raise ValueError("mpi ABC PMC not supported yet... ")
                self.pool.close()
            self.abcpmc = abcpmc
            self.threshold = self.read_ini("threshold",str, 'LinearEps')
            self.epimax = self.read_ini('epimax', float,5.0)
            self.epimin = self.read_ini('epimin',float, 1.0)
            self.part_prop = self.read_ini("particle_prop",str,'weighted_cov')
            self.set_prior = self.read_ini("set_prior",str,'Gaussian')
            self.knn = self.read_ini("num_nn",int, 10)
            self.npart = self.read_ini("npart",int,100)
            self.niter = self.read_ini("niter",int,2)
            self.ngauss = self.read_ini("ngauss",int,4)
            self.run_multigauss = self.read_ini("run_multigauss",bool,False)
            self.ndim = len(self.pipeline.varied_params)

            #options for decreasing threshold
            if self.threshold == 'ConstEps':
                self.eps = self.abcpmc.ConstEps(self.niter, self.epimax)
            elif self.threshold == 'ExpEps':
                self.eps = self.abcpmc.ExponentialEps(self.niter, self.epimax,self.epimin)
            else:
                self.eps = self.abcpmc.LinearEps(self.niter, self.epimax, self.epimin)

            print("\nRunning ABC PMC")
            print("with %d particles, %s prior, %s threshold, %d iterations over (%f,%f), %s kernal \n" % (self.npart,self.set_prior,self.threshold,self.niter,self.epimax,self.epimin,self.part_prop))

            #Data file is read for use in dist() for each step
            #parameter covariance used in the prior   
            if self.run_multigauss:
                self.sigma,self.data = self.generate_data(5000,0.25,1.)
            else:
                self.data = self.load_data()
                self.sigma = self.load_covariance_matrix()
        
            #At the moment the same prior (with variable hyperparameters) is
            # used for all parameters  - would be nice to change this to be more flexible 
            self.pmin = np.zeros(self.ndim)
            self.pmax = np.zeros(self.ndim)
            for i,pi in enumerate(self.pipeline.varied_params):
                self.pmin[i] = pi.limits[0]
                self.pmax[i] = pi.limits[1]
            self.p0 = np.array([param.start for param in self.pipeline.varied_params])
            if self.set_prior == 'uniform':
                self.prior = self.abcpmc.TophatPrior(self.pmin,self.pmax)
            else:
                self.prior = self.abcpmc.GaussianPrior(self.p0, self.sigma*2) 

            #create sampler
            self.sampler = self.abcpmc.Sampler(N=self.npart, Y=self.data, postfn=abc_model, dist=self.dist,pool=self.pool)

            #set particle proposal kernal
            self.abcpmc.Sampler.particle_proposal_kwargs = {}
            if self.part_prop == 'KNN':
                self.abcpmc.Sampler.particle_proposal_kwargs = {'k':self.knn}
                self.sampler.particle_proposal_cls = self.abcpmc.KNNParticleProposal
            elif self.part_prop == 'OLCM':
                self.sampler.particle_proposal_cls = self.abcpmc.OLCMParticleProposal

            self.converged = False

    def output_samples(self,out):
        for o in out:
            self.output.parameters(o)

    def execute(self):
        # called only by master. Other processors will be in self.pool
        # generate outputs somehow by running log_probability_function
        outputs = []
        for output in self.sampler.sample(self.prior, self.eps):
            if self.run_multigauss:
                print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(output.t, output.eps, output.ratio))
                for i, (mean, std) in enumerate(zip(np.mean(output.thetas, axis=0), np.std(output.thetas, axis=0))):
                    print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))
            outputs.append(np.c_[output.thetas,output.ws,output.dists])
        for out in outputs:
            self.output_samples(out)

        self.converged = True


    def is_converged(self):
        return  self.converged

     #distance function: sum of abs mean differences
        #Could add other distance functions here -EJ
    def dist(self,x, y):
        return np.sum(np.abs(np.mean(x, axis=0) - np.mean(y, axis=0)))


    def load_covariance_matrix(self):
        covmat_filename = self.read_ini("covmat", str, "").strip()
        if covmat_filename == "":
            covmat = np.array([p.width()/100.0 for p in self.pipeline.varied_params])
        elif not os.path.exists(covmat_filename):
            raise ValueError(
            "Covariance matrix %s not found" % covmat_filename)
        else:
            covmat = np.loadtxt(covmat_filename)

        if covmat.ndim == 0:
            covmat = covmat.reshape((1, 1))
        elif covmat.ndim == 1:
            covmat = np.diag(covmat ** 2)

        nparams = len(self.pipeline.varied_params)
        if covmat.shape != (nparams, nparams):
            raise ValueError("The covariance matrix was shape (%d x %d), "
                    "but there are %d varied parameters." %
                    (covmat.shape[0], covmat.shape[1], nparams))
        return covmat

    def load_data(self):
        data_filename = self.read_ini("datafile", str, "").strip()
        if not os.path.exists(data_filename):
            raise ValueError("Data file %s not found" % data_filename)
        else:
            self.data = np.loadtxt(data_filename)

    def generate_data(self,size,std,Max):
        sigma = np.eye(self.ngauss) * std
        means = Max*np.random.random_sample((self.ngauss,))
        if self.run_multigauss: print("True means: " ,means)
        np.savetxt("abc_multigauss_means.txt",np.vstack(means))
        data = np.random.multivariate_normal(means, sigma, size)
        return sigma,data
