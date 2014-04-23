from sampler import ParallelSampler
import numpy as np

INI_SECTION = "gridmax"

def task(p):
    return pipeline.posterior(p)


class GridMaxSampler(ParallelSampler):

    def config(self):
        global pipeline
        pipeline = self.pipeline

        if self.is_master():
            self.nsteps = self.ini.getint(INI_SECTION, "nsteps", 24)
            self.tolerance = self.ini.getfloat(INI_SECTION, "tolerance", 0.1)
            self.ndim = len(self.pipeline.varied_params)
            self.p = self.pipeline.normalize_vector(self.pipeline.start_vector())
            self.dimension = 0
            self.bounds = [(0,1) for i in xrange(self.ndim)]
            self.previous_maxlike = -np.inf
            self.maxlike = -np.inf


    def execute(self):
        #Do a slice maximization through the current
        #dimension, with the current bounds, in parallel
        #Start by setting all the points to the current one

        normed_points = [self.p.copy() for i in xrange(self.nsteps)]

        #Figure out which points to sample at in the current 
        #dimension
        d = self.dimension
        bounds = self.bounds[d]
        samples = np.linspace(bounds[0], bounds[1], self.nsteps)

        self.output.log_noisy("Minimizing in %s"%self.pipeline.varied_params[d].name)

        #Fill in the sample points for the current
        #dimension in the samples
        for (p,s) in zip(normed_points,samples):
            p[d] = s
        # print 'normed points', normed_points
        #Denormalize the points to get the physical parameters
        points = [self.pipeline.denormalize_vector(p) for p in normed_points]

        #And send them off to the workers
        if self.pool:
            results = self.pool.map(task, points)
        else:
            results = map(task, points)

        #Log the results for posterity
        for p, (_, e) in zip(points, results):
            self.output.parameters(p, e)

        #And now update our information.
        #We need to find the two points either side
        #of the maximum likelihood and set them as our new bound,
        #and set our new starting point to be the max value
        posteriors = np.array([p for (p,e) in results])
        best = posteriors.argmax()
        #first, the special cases one of the boundary 
        #points is the best (this might be cause for concern, of course)
        if best==0:
            #If we are up against the lower boundary then
            #use the edge as the lower bound, the first point 
            #as the upper, and the half-way point as the new start
            low = 0.0
            high = normed_points[1][d]
            start = high/2.0
        elif best==self.nsteps-1:
            #if on the upper edge, do the mirror image
            low = normed_points[-1][d]
            high = 1.0
            start = (normed_points[-1][d] + 1.0) / 2.0
        else:
            #but the usual case is to just bracket the
            #best-fit point
            low = normed_points[best-1][d]
            high = normed_points[best+1][d]
            start = normed_points[best][d]

        #update the bounds and start with whatever we found
        self.p[d] = start
        self.bounds[d] = (low, high)
    
        #If we have been around all the parameters 
        #then update the best-fits
        if d==0:
            self.previous_maxlike = self.maxlike
            self.maxlike = results[best][0]

        self.output.log_noisy("New best fit L = %lf at %s = %lf"%(posteriors.max(), self.pipeline.varied_params[d].name,points[best][d]))

        #and go on to the next dimension
        self.dimension = (d+1)%self.ndim

    def is_converged(self):
        return self.dimension==1 and (0<self.maxlike-self.previous_maxlike<self.tolerance)
