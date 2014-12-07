from .. import ParallelSampler
import numpy as np
from snake import Snake

def posterior(p_in):
    #Check the normalization
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
        print p_in
        return -np.inf, [np.nan for i in xrange(len(snake_pipeline.extra_saves))]
    p = snake_pipeline.denormalize_vector(p_in)
    like, extra = snake_pipeline.posterior(p)
    return like, extra


class SnakeSampler(ParallelSampler):
    sampler_outputs = [("like", float)]
    parallel_output = False

    def config(self):
        global snake_pipeline
        snake_pipeline=self.pipeline
        if self.is_master():
            self.threshold = self.read_ini("threshold", float, 6.0)
            self.grid_size = 1.0/self.read_ini("nsample_grid", float, 0.01)
            self.maxiter = self.read_ini("maxiter", int, 100000)

            
            origin = self.pipeline.normalize_vector(self.pipeline.start_vector())
            spacing = np.repeat(self.grid_size, len(self.pipeline.varied_params))
            self.snake = Snake(posterior, origin, spacing, pool=self.pool)

    def execute(self):
        X, P, E = self.snake.iterate()
        for (x,post,extra) in zip(X,P,E):
            try:
                x = self.pipeline.denormalize_vector(x)
                self.output.parameters(x, extra, post)
            except ValueError:
                print "The snake is trying to escape its bounds!"



    def is_converged(self):
        return self.snake.converged() or self.snake.iterations > self.maxiter
