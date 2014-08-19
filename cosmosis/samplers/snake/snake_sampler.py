from .. import Sampler
import numpy as np
from snake import Snake

class SnakeSampler(Sampler):
    sampler_outputs = [("like", float)]

    def config(self):
        self.threshold = self.read_ini("threshold", float, 6.0)
        self.grid_size = self.read_ini("grid_size", float, 0.01)
        self.maxiter = self.read_ini("maxiter", int, 100000)

        def posterior(p_in):
            #Check the normalization
            if (not np.all(p_in>=0)) or (not np.all(p_in<=1)):
                print p_in
                return -np.inf, [np.nan for i in xrange(len(self.pipeline.extra_saves))]
            p = self.pipeline.denormalize_vector(p_in)
            like, extra = self.pipeline.posterior(p)
            return like, extra
        
        origin = self.pipeline.normalize_vector(self.pipeline.start_vector())
        spacing = np.repeat(self.grid_size, len(self.pipeline.varied_params))

        self.snake = Snake(posterior, origin, spacing)

    def execute(self):
        x, post, extra = self.snake.iterate()
        try:
            x = self.pipeline.denormalize_vector(x)
            self.output.parameters(x, extra, post)
        except ValueError:
            print "The snake is trying to escape its bounds!"



    def is_converged(self):
        return self.snake.converged()
