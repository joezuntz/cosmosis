from .utils import std_weight
import numpy as np

def smooth_density_estimate_1d(samples, xmin, xmax, weights=None, N=1024, smoothing=1, fix_boundary=True):
    """
    Generate a smooth estimate of a 1D PDF from some samples using Kernel Density Estimation,
    
    The boundary fix algorithm comes from Jones & Foster 1996, by way of the getdist code,
    which this implementation follows.  Like that code, it uses convolutions to do the main calculations.
    Without the fix, parameters whose posterior has a hard boundary that cuts off the PDF
    where it is still large will have a spurious decrease in the smoothed PDF.
    
    If the boundary fix is applied, the upper and lower limit are assumed to be
    hard cut-offs in the PDF, not just the range where you happen to want the plots.
    
    Parameters
    ----------
    samples: 1D array
        samples from the PDF
    xmin: float
        lower bound, used for both the choice of output range and the boundary fix (if active)
    xmax: float
        upper bound, ditto
    weights: array
        optiona, default None. Weights per sample
    N: int
        optional, default 1024.  Number of sample points to return.
        Much faster if a power of 2
    smoothing: float/int
        optional, default 1.  Widening of the kernel, to smooth more
    fix_boundary: bool
        optional, default True.  Apply the fix to the boundaries
    """
    from scipy.signal import fftconvolve

    # Compute the things we need for the bandwidth choice:
    if weights is None:
        neff = len(samples)
        stdev = samples.std()
    else:
        neff = weights.sum() ** 2 / (weights**2).sum()
        stdev = std_weight(samples, weights)

    # This is a standard factor for 1D KDEs
    scott_factor = neff**(-0.2)

    # We optionally allow additional smoothing specified by the user
    width = stdev * scott_factor * smoothing

    # First we get the raw histogram that forms the basis of
    # the rest of this.
    edges =  np.linspace(xmin, xmax, N+1)
    x = 0.5 * (edges[1:] + edges[:-1])
    P_histogram, _ = np.histogram(samples, bins=edges, weights=weights)
    w = (xmax - xmin) / N

    # smoothing scale in units of the bin width
    s = width / w

    # Make the Gaussian kernel with which we are convolving.
    # We go out to 3 sigma
    window_width = int(3 * s)
    window_x = np.arange(-window_width, window_width+1)
    kernel = np.exp(-0.5 * (window_x/s)**2)
    
    # Generate the smoothed version.  If we do not need
    # the boundary smoothing then this is our final output
    P_smooth = fftconvolve(P_histogram, kernel, 'same')
    
    # If so, normalize and return it here
    if not fix_boundary:
        P_smooth /= P_smooth.sum() * (x[1] - x[0])        
        return x, P_smooth

    # Generate the mask, a top-hat which cuts off where the
    # boundaries are
    full_width = N + 2 * window_width
    mask = np.ones(full_width)
    mask[:window_width] = 0
    mask[window_width] = 0.5
    mask[-(window_width+1):] = 0
    mask[-(window_width+1)] = 0.5

    # Apply the correction, which uses
    # modified versions of the convolution kernel to
    # get better estimates of the edge region
    
    a0 = fftconvolve(mask, kernel, 'valid')
    
    # Avoid a divide-by-zero
    ix = np.nonzero(a0 * P_smooth)
    a0 = a0[ix]
    
    # Main correction calculation
    P_norm = P_smooth[ix] / a0
    xK = window_x * kernel
    x2K = xK * window_x
    a1 = fftconvolve(mask, xK, mode='valid')[ix]
    a2 = fftconvolve(mask, x2K, mode='valid')[ix]
    xP = fftconvolve(P_histogram, xK, mode='same')[ix]

    # Apply the correction, which is done inside the exponential
    # to keep everything positive.
    scaling = (P_smooth[ix] * a2 - xP * a1) / (a0 * a2 - a1 ** 2)
    P_final = P_smooth.copy()
    P_final[ix] = P_norm * np.exp(np.minimum(scaling / P_norm, 4) - 1)

    # Normalize and return
    P_final /= P_final.sum() * (x[1] - x[0])
    return x, P_final
