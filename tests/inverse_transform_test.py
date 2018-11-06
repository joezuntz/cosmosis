import numpy as np
from cosmosis.runtime.prior import InverseTransformPrior

if __name__=='__main__':

    from scipy.stats import norm
    from matplotlib import pyplot as plt

    x = np.linspace(-3,3,128)
    y = norm.pdf(x, loc=0, scale=1)
    y_c = norm.cdf(x, loc=0, scale=1)

    np.savetxt('tests/test_px.txt', np.column_stack([x, y]))

    prior = InverseTransformPrior('tests/test_px.txt')
    prior_trunc = prior.truncate(-1,1)

    x = np.linspace(prior.lower, prior.upper, 128)
    px = np.ones_like(x)
    for i in np.arange(len(x)):
        px[i] = np.exp(prior(x[i]))

    x_recovered = prior.denormalize_from_prior(y)
    print(prior)
    print(prior_trunc)
    
    samples = prior.sample(100000)
    samples_trunc = prior_trunc.sample(100000)

    plt.figure(figsize=(4.5, 3.75))
    plt.hist(samples, normed=True, bins=100, label='Samples')
    plt.hist(samples_trunc, normed=True, bins=100, label='Truncated Samples')
    plt.plot(x, px, 'o', label='Recovered PDF')
    plt.plot(x_recovered, y, 'o', label='Recovered CDF')
    plt.plot(x, y, label='Target PDF')
    plt.plot(x, y_c, label='Target CDF')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$P(x)$')
    plt.savefig('tests/test_normal.png', dpi=300, bbox_inches='tight')
