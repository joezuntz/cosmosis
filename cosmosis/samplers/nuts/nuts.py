"""
PinNUTS is not No-U-Turn-Sampling. 
PinNUTS is dynamic euclidean HMC with multinomial sampling,
but DEHMCMS doesn't sound like peanuts.

Licence
---------

The MIT License (MIT)

Copyright (c) 2012 Morgan Fouesneau
Copyright (c) 2019 Johannes Buchner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import numpy as np
from numpy import log, exp, sqrt
import tqdm

__all__ = ['pinnuts']


def leapfrog(theta, r, grad, epsilon, f):
    """ Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    theta: ndarray[float, ndim=1]
        initial parameter position

    r: ndarray[float, ndim=1]
        initial momentum

    grad: float
        initial gradient value

    epsilon: float
        step size

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    OUTPUTS
    -------
    thetaprime: ndarray[float, ndim=1]
        new parameter position
    rprime: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    logpprime: float
        new lnp
    """
    # make half step in r
    rprime = r + 0.5 * epsilon * grad
    # make new step in theta
    thetaprime = theta + epsilon * rprime
    #compute new gradient
    logpprime, gradprime = f(thetaprime)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime


def find_reasonable_epsilon(theta0, grad0, logp0, f):
    """ Heuristic for choosing an initial value of epsilon """
    epsilon = 1.
    r0 = np.random.normal(0., 1., len(theta0))

    # Figure out what direction we should be moving epsilon.
    _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
    # brutal! This trick make sure the step is not huge leading to infinite
    # values of the likelihood. This could also help to make sure theta stays
    # within the prior domain (if any)
    k = 1.
    while np.isinf(logpprime) or np.isinf(gradprime).any():
        k *= 0.5
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon * k, f)

    epsilon = 0.5 * k * epsilon

    # acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
    # a = 2. * float((acceptprob > 0.5)) - 1.
    logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))
    a = 1. if logacceptprob > np.log(0.5) else -1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    # while ( (acceptprob ** a) > (2. ** (-a))):
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2. ** a)
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
        # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
        logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))

    print("find_reasonable_epsilon=", epsilon)

    return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def build_tree(theta, r, grad, v, j, epsilon, f, joint0):
    """The main recursion."""
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
        jointprime = logpprime - 0.5 * np.dot(rprime, rprime.T)
        # Is the simulation wildly inaccurate?
        sprime = jointprime - joint0 > -1000
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        logptree = jointprime - joint0
        #logptree = logpprime
        # Compute the acceptance probability.
        alphaprime = min(1., np.exp(jointprime - joint0))
        #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alphaprime, nalphaprime, logptree = build_tree(theta, r, grad, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if sprime:
            if v == -1:
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, sprime2, alphaprime2, nalphaprime2, logptree2 = build_tree(thetaminus, rminus, gradminus, v, j - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, sprime2, alphaprime2, nalphaprime2, logptree2 = build_tree(thetaplus, rplus, gradplus, v, j - 1, epsilon, f, joint0)
            # Conpute total probability of this trajectory
            logptot = np.logaddexp(logptree, logptree2)
            # Choose which subtree to propagate a sample up from.
            if np.log(np.random.uniform()) < logptree2 - logptot:
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            logptree = logptot
            # Update the stopping criterion.
            sprime = sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alphaprime, nalphaprime, logptree

def tree_sample(theta, logp, r0, grad, epsilon, f, joint, maxheight=np.inf):
    # initialize the tree
    # Resample u ~ uniform([0, exp(joint)]).
    # Equivalent to (log(u) - joint) ~ exponential(1).
    #logu = float(joint - np.random.exponential(1, size=1))

    thetaminus = theta
    thetaplus = theta
    rminus = r0[:]
    rplus = r0[:]
    gradminus = grad[:]
    gradplus = grad[:]
    logptree = 0

    j = 0  # initial heigth j = 0
    s = 1  # Main loop: will keep going until s == 0.

    while (s == 1 and j < maxheight):
        # Choose a direction. -1 = backwards, 1 = forwards.
        v = int(2 * (np.random.uniform() < 0.5) - 1)

        # Double the size of the tree.
        if (v == -1):
            thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, sprime, alpha, nalpha, logptree2 = build_tree(
                thetaminus, rminus, gradminus, v, j, epsilon, f, joint)
        else:
            _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, sprime, alpha, nalpha, logptree2 = build_tree(
                thetaplus, rplus, gradplus, v, j, epsilon, f, joint)

        # Use Metropolis-Hastings to decide whether or not to move to a
        # point from the half-tree we just generated.
        logptot = np.logaddexp(logptree, logptree2)
        if sprime and np.log(np.random.uniform()) < logptree2 - logptot:
            print("Accepting jump to logp =", logpprime, " Probabilliity was:", np.exp(logptree2 - logptot))
            logp = logpprime
            grad = gradprime[:]
            theta = thetaprime
        else:
            print("Rejecting jump to logp =", logpprime, " Probabilliity was:", np.exp(logptree2 - logptot))
        
        logptree = logptot
        
        # Decide if it's time to stop.
        s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
        # Increment depth.
        j += 1
    #print("jumping to:", theta)
    return alpha, nalpha, theta, grad, logp


class PinNUTS:
    def __init__(self, f, D, delta=0.6, epsilon=None):
        self.f = f
        self.epsilon = epsilon
        self.gamma = 0.05
        self.delta = delta
        self.t0 = 10
        self.kappa = 0.75
        self.mu = None
        self.Hbar = 0.0
        self.epsilonbar = 1.0
        self.D = D
        self.tuning_steps_taken = 0
        self.tuned = False

    def adapt_epsilon(self, alpha, nalpha):
        if self.tuned:
            raise ValueError("Cannot resume tuning once it has finished")
        self.tuning_steps_taken += 1
        eta1 = 1. / float(self.tuning_steps_taken + self.t0)
        self.Hbar = (1. - eta1) * self.Hbar + eta1 * (self.delta - alpha / float(nalpha))
        self.epsilon = exp(self.mu - sqrt(self.tuning_steps_taken) / self.gamma * self.Hbar)

        eta2 = self.tuning_steps_taken ** -self.kappa
        self.epsilonbar = exp((1. - eta2) * log(self.epsilonbar) + eta2 * log(self.epsilon))

    def end_adaptation(self):
        self.tuned = True
        self.epsilon = self.epsilonbar

    def sample(self, start, M, Madapt):
        theta = start
        logp, gradp = self.f(theta)

        for i in tqdm.trange(Madapt):
            theta, logp, gradp = self.one_sample(theta, logp, gradp, tune=True)
        if M == 0:
            return np.array([theta]), np.array([logp])
        samples = np.empty((M, self.D), dtype=float)
        samples_logp = np.empty(M, dtype=float)

        for i in tqdm.trange(M):
            theta, logp, gradp = self.one_sample(theta, logp, gradp)
            samples[i] = theta
            samples_logp[i] = logp

        return samples, samples_logp
    
    def set_initial_epsilon(self, theta0, logp, grad):
        self.epsilon = find_reasonable_epsilon(theta0, grad, logp, self.f)
        self.mu = log(10. * self.epsilon)        

    def one_sample(self, current_sample, current_logp, current_grad, tune=False):
        # Resample momenta.
        r0 = np.random.normal(0, 1, self.D)

        #joint lnp of theta and momentum r
        joint = current_logp - 0.5 * np.dot(r0, r0.T)

        if self.epsilon is None:
            self.set_initial_epsilon(current_sample, current_logp, current_grad)


        # if all fails, the next sample will be the previous one        
        alpha, nalpha, thetaprime, grad, logp = tree_sample(current_sample, current_logp, r0, current_grad, self.epsilon, self.f, joint, maxheight=10)

        if tune:
            # Do adaptation of epsilon if we're still doing burn-in.
            self.adapt_epsilon(alpha, nalpha)
        else:
            # Adaptation is complete
            self.end_adaptation()
        
        return thetaprime, logp, grad
    

    



def pinnuts(f, M, Madapt, theta0, delta=0.6, epsilon=None):
    """
    Implements the multinomial Euclidean Hamiltonian Monte Carlo sampler
    described in Betancourt (2016).

    Runs Madapt steps of burn-in, during which it adapts the step size
    parameter epsilon, then starts generating samples to return.

    Note the initial step size is tricky and not exactly the one from the
    initial paper.  In fact the initial step size could be given by the user in
    order to avoid potential problems

    INPUTS
    ------
    epsilon: float
        step size
        see nuts8 if you want to avoid tuning this parameter

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    M: int
        number of samples to generate.

    Madapt: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size epsilon.

    theta0: ndarray[float, ndim=1]
        initial guess of the parameters.

    KEYWORDS
    --------
    delta: float
        targeted acceptance fraction

    OUTPUTS
    -------
    samples: ndarray[float, ndim=2]
    M x D matrix of samples generated by NUTS.
    note: samples[0, :] = theta0
    """

    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    lnprob = np.empty(M + Madapt, dtype=float)

    logp, grad = f(theta0)
    samples[0, :] = theta0
    lnprob[0] = logp

    # Choose a reasonable first epsilon by a simple heuristic.
    if epsilon is None:
        epsilon = find_reasonable_epsilon(theta0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    Hbar = 0

    for m in tqdm.trange(1, M + Madapt):
        # Resample momenta.
        r0 = np.random.normal(0, 1, D)

        #joint lnp of theta and momentum r
        joint = logp - 0.5 * np.dot(r0, r0.T)

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]
        
        alpha, nalpha, thetaprime, grad, logp = tree_sample(samples[m - 1, :], lnprob[m - 1], r0, grad, epsilon, f, joint, maxheight=10)
        samples[m, :] = thetaprime[:]
        lnprob[m] = logp

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar
    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    return samples, lnprob, epsilon
