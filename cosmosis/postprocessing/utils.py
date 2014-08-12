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
	return np.interp(0.5, wc, x)