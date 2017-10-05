import numpy as np

def std_weight(x, w):
    mu = mean_weight(x,w)
    r = x-mu
    return np.sqrt((w*r**2).sum() / w.sum())

def mean_weight(x, w):
    return (x*w).sum() / w.sum()

def median_weight(x, w):
	a = np.argsort(x)
	w = w[a]
	x = x[a]
	wc = np.cumsum(w)
	wc/=wc[-1]
	return np.interp(0.5, wc, x)

def percentile_weight(x, w, p):
	a = np.argsort(x)
	w = w[a]
	x = x[a]
	wc = np.cumsum(w)
	wc/=wc[-1]
	return np.interp(p/100., wc, x)



def find_asymmetric_errorbars(levels, v, weights=None):
    from ..plotting.kde import KDE
    import scipy.optimize
    N = len(v)

    #Generate and normalize weights
    if weights is None:
        weights = np.ones(N)
    weights = weights / weights.sum()

    #Normalize the parameter values
    mu = mean_weight(v,weights)
    sigma = std_weight(v,weights)
    x = (v-mu)/sigma

    #Build the P(x) estimator
    K=KDE(x, weights=weights)

    #Generate the axis over which get P(x)
    xmin = x[weights>0].min()
    xmax = x[weights>0].max()
    X = np.linspace(xmin,xmax,500)
    Y = K.normalize_and_evaluate(np.atleast_2d(X))
    Y/=Y.max()

    peak1d = X[Y.argmax()]
    peak1d = peak1d*sigma+mu

    #Take the log but suppress the log(0) warning
    old_settings = np.geterr()
    np.seterr(all='ignore')
    Y=np.log(Y)
    np.seterr(**old_settings)  # reset to default


    #Calculate the levels
    def objective(level, target_weight):
        w = np.where(Y>level)[0]
        if len(w) == 0:
            weight_inside = 0.0
        else:
            low = X[w].min()
            high = X[w].max()
            inside = (x>=low) & (x<=high)
            weight_inside = weights[inside].sum()
        return (weight_inside - target_weight)

    limits = []
    for target_weight in levels:
        level = scipy.optimize.bisect(objective, Y[np.isfinite(Y)].min(), Y.max(), args=(target_weight,))
        w = np.where(Y>level)[0]
        low = X[w].min()
        high = X[w].max()
        #Convert back to origainal space
        low = low*sigma+mu
        high = high*sigma+mu
        limits.append((low,high))

    return peak1d, limits
