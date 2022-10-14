import numpy as np
from tqdm import tqdm
import george
import emcee
from george import kernels
from multiprocessing import Pool, cpu_count
from scipy.optimize import fmin_powell, minimize
import scipy.optimize as op

# def gp_ln_likelihood(hyper_pars, theta, scalar, ndim):  
#     a = hyper_pars[0]
#     b = hyper_pars[1:]
#     if (a < 1e-7):
#         return 1e+9
#     for b_i in b:
#         if (b_i > 1e+6) or (b_i < 1e-7):
#             return 1e+9
#     K = a * kernels.ExpSquaredKernel(b, ndim=ndim)
#     gp = george.GP(K)
#     gp.compute(theta)
#     return -gp.lnlikelihood(scalar)

# def gp_grad(hyper_pars, theta, scalar, ndim):
#     a = hyper_pars[0]
#     b = hyper_pars[1:]
#     if (a < 1e-7):
#         return np.zeros(ndim + 1)
#     for b_i in b:
#         if (b_i > 1e+6) or (b_i < 1e-7):
#             return np.zeros(ndim + 1)
#     K = a * kernels.ExpSquaredKernel(b, ndim=ndim)
#     gp = george.GP(K)
#     gp.compute(theta)
#     return -gp.grad_log_likelihood(scalar)


def gp_ln_likelihood_and_grad(hyper_pars, theta, scalar, ndim):  
    a = hyper_pars[0]
    b = hyper_pars[1:]
    if (a < 1e-7):
        return 1e+9, np.zeros(ndim + 1)
    for b_i in b:
        if (b_i > 1e+6) or (b_i < 1e-7):
            return 1e+9, np.zeros(ndim + 1)
    K = a * kernels.ExpSquaredKernel(b, ndim=ndim)
    gp = george.GP(K)
    gp.compute(theta)
    log_like = -gp.lnlikelihood(scalar)
    grad = -gp.grad_log_likelihood(scalar)
    return log_like, grad


class GPEmulator:
    def __init__(self, N_DIM, OUTPUT_DIM, dv_fid, dv_std):
        self.N_DIM      = N_DIM
        self.OUTPUT_DIM = OUTPUT_DIM
        self.dv_fid = dv_fid
        self.dv_std = dv_std

    def train_gp_i(self, i):
        init_hp_vals = np.ones(self.N_DIM + 1)
        init_hp_vals[0] = 0.5
        method="BFGS"
        # method="L-BFGS-B"
        results = op.minimize(gp_ln_likelihood_and_grad, init_hp_vals, jac=True, 
                                method=method, args=(self.fit_theta_norm, self.vector_norm[:,i], self.N_DIM))
        opt_hp_val = results.x
        return opt_hp_val

    def train(self, fit_theta, vector):
        print("Finding optimal GP hyperparameters...")
                
        # Normalize the scalars
        self.vector_mean = self.dv_fid
        self.vector_std  = self.dv_std
        self.vector_norm = (vector - self.vector_mean) / self.vector_std
        
        self.theta_min = fit_theta.min(axis=0)
        self.theta_max = fit_theta.max(axis=0)
        self.fit_theta_norm = (fit_theta - self.theta_min) / (self.theta_max - self.theta_min)

        opt_hp_list = []
        gp_list = []
        
        i = 0

        print("JAZ removed parallelism - reinstate!")    
        results = list(tqdm(map(self.train_gp_i, np.arange(self.OUTPUT_DIM)), total=self.OUTPUT_DIM))
        #results = p.map(self.train_gp_i, np.arange(self.OUTPUT_DIM)) 
        for result in results:
            opt_hp_list.append(result)

        for i, opt_hp_val in enumerate(opt_hp_list):
            K  = opt_hp_val[0] * kernels.ExpSquaredKernel(opt_hp_val[1:], ndim=self.N_DIM)
            gp = george.GP(K)
            gp.compute(self.fit_theta_norm)
            gp_list.append(gp)
            
        
        self.opt_hp_list = opt_hp_list
        self.trained_gp = gp_list
        self.trained = True
        print("Found optimal GP hyperparameters:")
        print(opt_hp_list)

    def predict(self, theta_pred):
        assert self.trained, "The emulator needs to be trained first before predicting"
        theta_pred_norm = (theta_pred - self.theta_min) / (self.theta_max - self.theta_min)
        dv_pred = np.zeros((theta_pred.shape[0], self.OUTPUT_DIM))
        for i in range(self.OUTPUT_DIM):
            gp = self.trained_gp[i]
            scalar_pred = gp.predict(self.vector_norm[:,i], theta_pred_norm, return_cov=False)
            dv_pred[:,i] = scalar_pred * self.vector_std[i] + self.vector_mean[i]
        return dv_pred    
    
    def save(self, filename):
        assert self.trained, "GP not trained!"
        print("Saving GP model...")
        with h5.File(filename, 'w') as f:
            f['opt_hp']         = np.array(self.opt_hp_list)
            f['vector_mean']    = self.vector_mean
            f['vector_std']     = self.vector_std
            f['vector_norm']    = self.vector_norm
            f['theta_min']      = self.theta_min
            f['theta_max']      = self.theta_max
            f['fit_theta_norm'] = self.fit_theta_norm
            
    def load(self, filename):
        print("Loading trained GP model...")
        with h5.File(filename, 'r') as f:
            opt_hp_list = f['opt_hp'][:]
            self.opt_hp_list = opt_hp_list.tolist()
            self.vector_norm = f['vector_norm'][:]
            self.vector_mean = f['vector_mean'][:]    
            self.vector_std  = f['vector_std'][:]     
            self.theta_min   = f['theta_min'][:]
            self.theta_max   = f['theta_max'][:]
            self.fit_theta_norm = f['fit_theta_norm'][:]
        gp_list = []
        for i in range(self.OUTPUT_DIM):
            opt_hp_val = self.opt_hp_list[i]
            K  = opt_hp_val[0] * kernels.ExpSquaredKernel(opt_hp_val[1:], ndim=self.N_DIM)
            gp = george.GP(K)
            gp.compute(self.fit_theta_norm)
            gp_list.append(gp)
        self.trained_gp = gp_list
        self.trained = True