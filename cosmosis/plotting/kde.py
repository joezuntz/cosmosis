from builtins import zip
from builtins import range
import scipy.stats
import numpy as np
from functools import wraps

def not_with_weights(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        self=args[0]
        if self.weights is not None:
            raise ValueError("The standard KDE functions have not been updated to use weights on points")
    return wrapper


class KDE(scipy.stats.kde.gaussian_kde):
    def __init__(self, points, factor=1.0, weights=None):
        points = np.array(np.atleast_2d(points))

        self._factor = factor
        d, n = points.shape
        self.norms = []
        self.weights=weights
        normalized_points = []
        #assert d<=2, "KDE is for 1D 2D only"
        for column in points:
            col_mean = column.mean()
            col_std = column.std()
            self.norms.append((col_mean,col_std))
            normalized_points.append((column-col_mean)/col_std)
        super(KDE,self).__init__(normalized_points)

    def covariance_factor(self):
        return self.scotts_factor() * self._factor

    def grid_evaluate(self, n, ranges):
        if isinstance(ranges,tuple):
            ranges = [ranges]
        slices = [slice(xmin,xmax,n*1j) for (xmin,xmax) in ranges]
        grids = np.mgrid[slices]
        axes = [ax.squeeze() for ax in np.ogrid[slices]]
        flats = [(grid.flatten()-norm[0])/norm[1]
                 for (grid,norm) in zip(grids,self.norms)]

        shape = grids[0].shape
        flats = np.array(flats)
        like_flat = self.evaluate(flats)
        like = like_flat.reshape(*shape)
        if len(axes)==1:
            axes = axes[0]
        return axes,like

    def normalize_and_evaluate(self, points):
        points = np.array([(p-norm[0])/norm[1] for norm, p in zip(self.norms, points)])
        return self.evaluate(points)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        JAZ: Overriding this function to allow the use of weights.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = np.atleast_2d(points)
        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = np.zeros((m,), dtype=np.float)


        if m >= self.n:
            weights = self.weights
            if weights is None: weights=np.repeat(1.0,self.n)
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff*tdiff,axis=0) / 2.0
                result = result + weights[i]*np.exp(-energy)
        else:
            weights = self.weights
            if weights is None: weights=1

            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(weights*np.exp(-energy), axis=0)/np.sum(weights)

        result = result / self._norm_factor

        return result

    @not_with_weights
    def integrate_gaussian(self, mean, cov):
        return super(KDE,self).integrate_gaussian(mean, cov)

    @not_with_weights
    def integrate_box_1d(self, low, high):
        return super(KDE,self).integrate_box_1d(low, high)

    @not_with_weights
    def integrate_box(self, low_bounds, high_bounds, maxpts=None):
        return super(KDE,self).integrate_box(low_bounds, high_bounds, maxpts=maxpts)

    @not_with_weights
    def integrate_kde(self, other):
        return super(KDE, self).integrate_kde(other)

    @not_with_weights
    def resample(self):
        return super(KDE, self).resample()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().

        JAZ overridden to use weights
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            #JAZ this is the overridden bit
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, aweights=self.weights, bias=False))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) * self.n
