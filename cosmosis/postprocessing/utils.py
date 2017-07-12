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



def find_asymmetric_errorbars(levels, x, weights=None):
    from ..plotting.kde import KDE
    import scipy.optimize
    N = len(x)

    #Normalize weights
    if weights is None:
        weights = np.ones(N)

    weights = weights / weights.sum()

    K=KDE(x, weights=weights)
    xmean = np.average(x, weights=weights)
    xmin = x[weights>0].min()
    xmax = x[weights>0].max()

    ymax = K.evaluate(xmean)[0]

    X = np.linspace(xmin,xmax,500)
    Y = K.evaluate(X)

    def objective(level, target_level):
        w = np.where(Y>level)[0]
        if len(w) == 0:
            weight_inside = 0
        else:
            low = X[w].min()
            high = X[w].max()
            inside = (x>=low) & (x<=high)
            weight_inside = weights[inside].sum()
        return weight_inside - target_level

    limits = []
    for target_level in levels:
        level = scipy.optimize.bisect(objective, 0.0, 1.0, args=(target_level,))
        w = np.where(Y1>level)[0]
        low = X[w].min()
        high = X[w].max()
        limits.append((low,high))
    return limits
