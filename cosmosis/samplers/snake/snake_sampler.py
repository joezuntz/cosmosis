from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import range
from .. import ParallelSampler
import numpy as np
from .snake import Snake

def posterior(p_in):
    #Check the normalization
    if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
        print(p_in)
        return -np.inf, [np.nan for i in range(len(snake_pipeline.extra_saves))]
    p = snake_pipeline.denormalize_vector(p_in)
    like, extra = snake_pipeline.posterior(p)
    return like, extra


class SnakeSampler(ParallelSampler):
    sampler_outputs = [("post", float)]
    parallel_output = False

    def config(self):
        global snake_pipeline
        snake_pipeline=self.pipeline
        if self.is_master():
            threshold = self.read_ini("threshold", float, 4.0)
            self.grid_size = 1.0/self.read_ini("nsample_dimension", int, 10)
            self.maxiter = self.read_ini("maxiter", int, 100000)

            
            origin = self.pipeline.normalize_vector(self.pipeline.start_vector())
            spacing = np.repeat(self.grid_size, len(self.pipeline.varied_params))
            self.snake = Snake(posterior, origin, spacing, threshold, pool=self.pool)

    def execute(self):
        X, P, E = self.snake.iterate()
        for (x,post,extra) in zip(X,P,E):
            try:
                x = self.pipeline.denormalize_vector(x)
                self.output.parameters(x, extra, post)
            except ValueError:
                print("The snake is trying to escape its bounds!")



    def is_converged(self):
        if self.snake.converged():
            print("Snake has converged!")
            print("Best post = %f    Best surface point = %f" %(self.snake.best_like_ever, self.snake.best_fit_like))
            return True
        if self.snake.iterations > self.maxiter:
            print("Run out of iterations.")
            print("Done %d, max allowed %d" % (self.snake.iterations, self.maxiter))
            return True
        return False
