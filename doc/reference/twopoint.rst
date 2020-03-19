Two-point function calculations
===============================

2D Spectra From 3D: Overview
--------------------------------

The CosmoSIS standard library module project_2d can be used to
integrate 3D power spectra (such as matter or galaxy power spectra)
with a pair of kernels into 2D power spectra using the Limber integral.

This is an approximation which often applies at small scales and when
using a broad enough kernel.

In a flat cosmology the Limber integral is:

.. math::
    C^{12}_\ell =  A \int_0^{\chi_1} W_1(\chi) W_2(\chi) P(k=(l+0.5)/\chi, z(\chi)) / chi^2 d\chi

where the two W functions are kernels that describe the response of the statistic to
with distance and P is a 3D power spectrum.  Different quantities can be calculated using different
choices for the W and P functions.

For galaxy clustering spectra, P is the galaxy power spectrum and
the W functions are the galaxy number densities n(z).

For weak lensing spectra, P is the matter power spectrum and the W functions
are given by:

.. math::
    W^{\mathrm{WL}}_\chi =  \frac{3}{2}\Omega_m H_0^2 a^{-1}(\chi) \chi \frac{1}{\bar{n}} \int_\chi^\infty \mathrm{d}\chi_s n(\chi_s) \frac{\chi_s - \chi}{\chi_s}

This quantitiy is calculated in the module.


2D Spectra From 3D: Basic Usage
--------------------------------



2D Spectra From 3D: Advanced Usage
--------------------------------



2D Correlation Functions From Spectra
-------------------------------------


2pt Likelihoods
---------------------------
