import numpy as np
import pdb


class FisherParameterError(Exception):
    def __init__(self, parameter_index):
        message = "Fisher Matrix likelihood function returned None for parameter: {}".format(parameter_index)
        super(Exception,self).__init__(self, message)
        self.parameter_index = parameter_index


class Fisher(object):
    def __init__(self, compute_vector, start_vector, step_size, tolerance, maxiter, pool=None):
        
        self.compute_vector = compute_vector
        self.maxiter = maxiter
        self.step_size = step_size
        self.start_params = start_vector
        self.current_params = start_vector
        self.nparams = start_vector.shape[0]
        self.iterations = 0
        self.pool = pool

    def converged(self):
        crit = (abs(self.new_onesigma - self.old_onesigma).max() < self.threshold)
        return crit

    def compute_derivatives(self):
        derivatives = []
        points = []

        #To improve parallelization we first gather all the data points
        #we use in all the dimensions
        for p in range(self.nparams):
            points +=  self.five_points_stencil_points(p)
        print("Calculating derivatives using {} total models".format(len(points)))
        if self.pool is None:
            results = list(map(self.compute_vector, points))
        else:
            results = self.pool.map(self.compute_vector, points)

        # Bit of a waste of time to compute the inv cov separately,
        # but it's a quick fix to a memory error if we compute the cov
        # for every single value
        _, inv_cov = self.compute_vector(points[0], cov=True)

        #Now get out the results that correspond to each dimension
        for p in range(self.nparams):
            results_p = results[4*p:4*(p+1)]
            derivative = self.five_point_stencil_deriv(results_p, p)
            derivatives.append(derivative)
        derivatives = np.array(derivatives)
        return derivatives, inv_cov


    def compute_fisher_matrix(self):
        derivatives, inv_cov = self.compute_derivatives()

        if not np.allclose(inv_cov, inv_cov.T):
            print("WARNING: The inverse covariance matrix produced by your pipeline")
            print("         is not symmetric. This probably indicates a mistake somewhere.")
            print("         If you are only using cosmosis-standard-library likelihoods please ")
            print("         open an issue about this on the cosmosis site.")
        fisher_matrix = np.einsum("il,lk,jk->ij", derivatives, inv_cov, derivatives)
        return fisher_matrix

    def five_points_stencil_points(self, param_index):
        delta = np.zeros(self.nparams)
        delta[param_index] = 1.0
        points = [self.current_params + x*delta for x in 
            [2*self.step_size, 
             1*self.step_size, 
            -1*self.step_size, 
            -2*self.step_size]
        ]
        return points        

    def five_point_stencil_deriv(self, obs, param_index):
        for r in obs:
            if r is None:
                raise FisherParameterError(param_index)
        deriv = (-obs[0] + 8*obs[1] - 8*obs[2] + obs[3])/(12*self.step_size)
        return deriv

    def compute_one_sigma(Fmatrix):
        sigma = np.sqrt(np.linalg.inv(Fmatrix))
        return sigma

class NumDiffToolsFisher(Fisher):
    def compute_derivatives(self):
        import numdifftools as nd
        def wrapper(param_vector):
            print("Running pipeline:", param_vector)
            return self.compute_vector(param_vector, cov=False)
        jacobian_calculator = nd.Jacobian(wrapper, step=self.step_size)
        derivatives = jacobian_calculator(self.current_params)
        _, inv_cov = self.compute_vector(self.current_params, cov=True)
        print(derivatives.shape, inv_cov.shape)
        return derivatives.T, inv_cov
    
class STEMFisher(Fisher):
    def compute_derivatives(self):
        derivatives = []
        for i in range(self.nparams):
            def wrapper(x):
                p = self.current_params.copy()
                p[i] = x
                return self.compute_vector(p, cov=False)
            calc = SarcevicDerivativeCalculator(wrapper, self.current_params[i], dx=self.step_size, pool=self.pool)
            derivatives.append(calc.stem_method())
        _, inv_cov = self.compute_vector(self.current_params, cov=True)
        return np.array(derivatives), inv_cov

class SarcevicDerivativeCalculator:
    """
    This is Niko Sarcevic's robust derivative calculator class, from
    https://github.com/nikosarcevic/Derivative-Calculator

    Thanks and credit to Niko!

    MIT License

    Copyright (c) 2023 Nikolina Sarcevic

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
    def __init__(self, myfunc, x_center, dx=0.01, min_samples=5, pool=None):
        """
        Initialize the Derivative Calculator with a function of a single parameter,
        a central value for evaluation, and parameters for derivative calculation
        and statistical analysis.

        Parameters:
        myfunc (callable): The function whose derivative is to be calculated.
        x_center (float): The central value at which the derivative is evaluated.
        dx (float): The step size for derivative calculation.
        iterations (int): Number of iterations for derivative calculation.
        min_samples (int): Minimum number of samples to retain while adjusting the range.
        """
        self.myfunc = myfunc
        self.x_center = x_center
        self.min_samples = min_samples
        self.dx = dx
        self.stem_deriv = None
        self.pool = pool


    def stem_method(self):
        """
        Calculates the stem derivative of a function at a given central value.

        The "stem" derivative is based on a method developed by Camera et al. as described
        in the paper "SKA Weak Lensing III: Added Value of Multi-Wavelength Synergies for
        the Mitigation of Systematics" (https://arxiv.org/abs/1606.03451).
        A detailed description of the method can be found in Appendix B of the paper.

        Note that this method is not applicable for functions with a zero derivative at
        the central value. It also may require a relatively long time to converge for
        functions with a very small derivative at the central value.

        JZ: Modified this code to use a pool object for parallelism.
        
        Returns:
        tuple: the stem derivative.
        """

        # The percentage values to use for the stem method
        # Note that this is an arbitrary choice and the values
        # can be changed as needed
        percentages = [0.00625, 0.0125, 0.01875, 0.025, 0.0375, 0.05,
                       0.1]  # 0.625%, 1.25%, 1.875%, 2.5%, 3.75%, 5%, 10%
        stem_deriv = []  # List to store the stem derivative

        # Use a fixed range around zero for x values if the central value is zero
        if self.x_center == 0:
            x_values = np.array([p for p in percentages]
                                + [-p for p in percentages])
        # Use a fixed range around the central value for x values otherwise
        else:
            x_values = np.array([self.x_center * (1 + p) for p in percentages]
                                + [self.x_center * (1 - p) for p in percentages])
        # Evaluate the function at these x values
        if self.pool:
            y_values = self.pool.map(self.myfunc, x_values)
        else:
            y_values = [self.myfunc(x) for x in x_values]
        y_values = np.stack(y_values, axis=0)

        # Fit a line to the data points and calculate the spread
        while len(x_values) >= self.min_samples:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            y_fitted = slope * x_values[:, None] + intercept
            spread = np.abs((y_fitted - y_values) / y_values)
            max_spread = np.max(spread)

            # If the spread is small enough, return the slope as the derivative
            # Also note that this criterium is an arbitrary choice
            # and the value can be changed as needed
            if max_spread < 0.01:
                stem_deriv.append(slope)
                break
            # Otherwise, remove the outlier point with the maximum spread
            else:
                x_values = x_values[1:-1]
                y_values = y_values[1:-1]

        return stem_deriv[0] if stem_deriv else 0  # Or return None

    # maybe use this later
    def stem_method_with_fitting(self, x_values, y_values):
        """
        Calculates the stem method with fitting using provided data points.
        It fits a line to the data points and removes the outlier points iteratively.
        The fitting is done using numpy.polyfit.
        Other fitting methods can be used instead.

        Parameters:
        x_values (array-like): x values of data points.
        y_values (array-like): y values of data points.

        Returns:
        tuple: Slope and intercept of the fitted line.
        """
        # Initial fitting
        slope, intercept = np.polyfit(x_values, y_values, 1)
        y_fitted = slope * x_values + intercept
        spread = np.abs((y_fitted - y_values) / y_values)  # Spread of the fitted line
        max_spread = np.max(spread)  # Maximum spread

        # Refinement with removal of outliers and re-fitting
        while max_spread >= 0.01 and len(x_values) >= self.min_samples:
            # Find the index of the point with maximum spread
            max_spread_index = np.argmax(spread)

            # Remove the outlier point
            x_values = np.delete(x_values, max_spread_index)
            y_values = np.delete(y_values, max_spread_index)

            # Fit the line again
            slope, intercept = np.polyfit(x_values, y_values, 1)
            y_fitted = slope * x_values + intercept
            spread = np.abs((y_fitted - y_values) / y_values)
            max_spread = np.max(spread)

        return slope, intercept, max_spread

def test():
    def theory_prediction(x, cov=False):
        #same number of data points as parameters here
        x = np.concatenate([x,x])
        theory = 2*x + 2
        inv_cov = np.diag(np.ones_like(x)**-1)
        if cov:
            return theory, inv_cov
        else:
            return theory

    best_fit_params = np.array([0.1, 1.0, 2.0, 4.0,])
    fisher_calculator = Fisher(theory_prediction, best_fit_params, 0.01, 0.0, 100)
    F = fisher_calculator.compute_fisher_matrix()
    print(F)
    return F

if __name__ == '__main__':
    test()