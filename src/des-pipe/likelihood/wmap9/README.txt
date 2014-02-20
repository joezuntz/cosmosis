This is the WMAP likelihood software.

This software is written in Fortran 90; it has been tested with the SGI MIPSpro,
NAG, and Intel Fortran 90 compilers.  It requires the following library:

- The cfitsio library.
	http://heasarc.nasa.gov/fitsio/fitsio.html

- A set of Cholesky factorization routines.  On the SGI systems, the SCSL
Science Library contains the relevant routines; the LAPACK library will provide
the routines on other systems.

To build it, you should edit the Makefile to support your current environment;
the provided Makefile shows several possible configurations.

Then:
make			# Builds the likelihood software library.
make check		# Builds and runs the test program

Differences on the order of 0(0.001) are normal between computer platforms.

Modification History:
- 2006-Mar-16 - Version 2.0 - Initial release.

- 2006-Mar-17 - Version 2.1 - Additional error reporting routines have been
                added to provide safe exit from the likelihood routine in the 
                event of errors.

- 2006-Oct-16 - Version 2.2 - Adjustments to the TT likelihood module and to
		the point source correction.  Note that this version no longer 
                requires installation of the HEALPix library.

- 2006-Oct-18 - Version 2.2.1 - Beam error module failures treated as warnings
                instead of errors. 

- 2006-Nov-24 - Version 2.2.2 - Implements a faster low-lT TT likelihood module,
                suggested by Jon Sievers of CITA.

- 2008-Mar-05 - Version 3.0 - The 5-year data initial release.  Note that this
		version has two options for computing the low-l TT likelihood
		(pixel and gibbs), and an option to compute the likelihood for 
		TB and EB spectra.

- 2008-Oct-20 - Version 3.1 - A patch was released to make the data directory
		selectable through an environment variable.  Thanks to T. Kinser
		of Lawrence Berkeley National Laboratory for this patch.

- 2009-Mar-5  - Version 3.2 - Loop optimization.

- 2010-Jan-25 - Version 4.0 - The 7-year data initial release.

- 2010-May-27 - Version 4.1 - Correction of two small bugs affecting high-l
		TT, TE, TB, and point source power spectra.

- 2012-Dec-20  - Version 5.0 - The 9-year data initial release.

See the CHANGES file for details describing each release.

===============================================================================
	Tarball contents.
===============================================================================

The tarball contains a data subdirectory that contains various input files. 
These input files are either ASCII text files containing columns of data,
or FITS files containing binary data.

The tarball also contains the following primary files:

-------------------------------------------------------------------------------
0) README.txt
-------------------------------------------------------------------------------
This descriptive file.

-------------------------------------------------------------------------------
1) test.f90
-------------------------------------------------------------------------------
A wrapper program shows you how to call the likelihood code and allow you to
run and test it. For the test data provided you should get the output stored in
output.std so you may compare your output to this with a simple diff command.

-------------------------------------------------------------------------------
2) WMAP_9yr_likelihood.F90  
-------------------------------------------------------------------------------
The central likelihood routine from which others are called.
- MASTER codes for TTTT (l=2-1000) and TETE (l=2-800) + a log determinant term for TETE 
- The option of using a pixel based likelihood for TT 2<=l<=30 instead of MASTER
- The option of using a gibbs-sampling likelihood for TT 2<=l<=32 instead of MASTER
- The option of using a pixel based likelihood for TE, EE and BB 2<=l<=24 (this
  substitutes for MASTER TETE 2<l<24).
- The option of using a pixel based likelihood for TE, TB, EE, BB and EB 2<=l<=24 (this
  substitutes for MASTER TETE 2<l<24).
- The option of using a master likelihood for TB (l=2-450) + a log determinant
  term for TBTB

-------------------------------------------------------------------------------
3) wmap_9yr_options.F90
-------------------------------------------------------------------------------
The options/parameters for use in the code are all contained the WMAP_options
module.

-------------------------------------------------------------------------------
4) WMAP_9yr_teeebb_pixlike.F90
-------------------------------------------------------------------------------
The low l TE/EE/BB combined pixel based analysis for l=2-23 

-------------------------------------------------------------------------------
5) WMAP_9yr_tetbeebbeb_pixlike.F90
-------------------------------------------------------------------------------
The low l TE/TB/EE/BB/EB combined pixel based analysis for l=2-23 

-------------------------------------------------------------------------------
6) WMAP_9yr_tt_beam_and_ptsrc_chisq.f90
-------------------------------------------------------------------------------
Extra chisq corrections in TTTT from the beams and point source uncertainties.

-------------------------------------------------------------------------------
7) WMAP_9yr_gibbs.F90
-------------------------------------------------------------------------------
The low l TT gibbs-sampling likelihood code for l<=32.

-------------------------------------------------------------------------------
8) WMAP_9yr_tt_pixlike.F90
-------------------------------------------------------------------------------
The low l TT pixel based likelihood code for l<=30.

-------------------------------------------------------------------------------
9) wmap_9yr_util.f90
-------------------------------------------------------------------------------
IO tools and error checker

-------------------------------------------------------------------------------
10) read_archive_map.f90, read_archive_map.fh
-------------------------------------------------------------------------------
A subroutine that read a skymap from a binary table in a FITS file.

-------------------------------------------------------------------------------
11) read_fits.f90, read_fits.fh
-------------------------------------------------------------------------------
A collections of subroutines that read two and three dimension floating point 
and complex arrays from the primary header/data unit of a FITS file.

-------------------------------------------------------------------------------
12) br_mod_dist.f90
-------------------------------------------------------------------------------
An F90 module for computing the Blackwell-Rao estimator given signal samples 
from the posterior [Written by Hans Kristian Eriksen] 
Used by the low l TT gibbs-sampling likelihood option.

-------------------------------------------------------------------------------
13) Makefile
-------------------------------------------------------------------------------
This is a standard makefile to compile the code.

-------------------------------------------------------------------------------
14) data/
-------------------------------------------------------------------------------
A directory containing data necessary to the software.

-------------------------------------------------------------------------------
15) CHANGES
-------------------------------------------------------------------------------
The list of changes made since the initial release.

