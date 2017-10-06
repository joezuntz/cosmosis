# Demo 15:  Galaxy two-point functions

This demo shows how to calculate a variety of two-point cosmological functions for a spectroscopic survey.


## Running ##

Run this demo with:

```
cosmosis demos/demo15.ini
```

and postprocess with 

```
postprocess demo_output_15 -o plots -p demo15
```

The files `plots/demo15_*.png` will contain figures showing cosmological quantites for these parameters, including the 2D matter power in a given bin, `plots/demo15_matter_power_2d.png:

![demo15_matter_power_2d.png](https://bitbucket.org/repo/KdA86K/images/3901544716-demo15_matter_power_2d.png)

and the slope of the galaxy luminosity function alpha, in `plots/demo15_galaxy_luminosity_slope.png`:

![demo15_galaxy_luminosity_slope.png](https://bitbucket.org/repo/KdA86K/images/3008458207-demo15_galaxy_luminosity_slope.png)

## Understanding ##

This demo builds on Demo 6 but shows a series of other galaxy two-point functions for a single photometric redshift bin.  Here's our pipeline:

```
#!ini
modules = consistency dndz luminosity_slope camb sigma8_rescale halofit_takahashi  angular_power 2pt_shear 2pt_matter 2pt_ggl 2pt_mag
```

In Demo 6 we loaded our n(z) function in tomographic bins from file.  In this example we will model it. Here it's just a simple Gaussian, but for realistic problems you can make this modelling as sophisticated as you need.

```
#!ini
[dndz]
file = cosmosis-standard-library/number_density/gaussian_window/gaussian_window.py
z = 1.0
sigma = 0.1
```


In demo 6 we just wanted shear, but now we tell the `angular_power` module which spectra we want it to generate using various True/False options.  We ask for shear, matter spectra, galaxy galaxy-lensing, magnification, and a cross spectrum:

```
#!ini
[angular_power]
file = cosmosis-standard-library/shear/spectra/interface.so
n_ell = 100
ell_min = 10.0
ell_max = 100000.0
shear_shear = T
intrinsic_alignments = F
matter_spectra = T
ggl_spectra = T
gal_IA_cross_spectra = F
mag_gal_cross_spectra = T
mag_mag = T
```

Since magnification also needs to know the slope of the luminosity function with redshift, we also have a module modelling that, too, based on a fitting function.

Finally we convert our spectra to more easily observable correlation functions.  Here we show an example of using the same module a number of times, but with different options each time:

```
#!ini
[2pt_matter]
file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so
input_section_name = matter_cl
output_section_name = matter_xi
; Type of Hankel transform and output correlation function
; [0 = shear, 1 = matter, 2 = ggl]
corr_type = 1

[2pt_ggl]
file = cosmosis-standard-library/shear/cl_to_xi_nicaea/nicaea_interface.so
input_section_name = ggl_cl
output_section_name = ggl_xi
corr_type = 2
```

This can be a very powerful mechanism for doing a collection of smaller pipeline tasks slightly differently. Also note that we are manually specifying the output section to use.  This can also be useful if you want to save results for multiple different surveys, for example.
