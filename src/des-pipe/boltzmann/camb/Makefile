include ../../../Makefile.mine
#include Makefile.mine

FITSDIR = $(CFITSIO_LIB)
FITSLIB = cfitsio

F90C = $(F90)
FFLAGS += -fPIC 

#Set FISHER=Y to compile bispectrum fisher matrix code
FISHER=
#FFLAGS += -Ofast -ffast-math -funroll-loops 
FFLAGS += -ffree-line-length-none

ifneq ($(FISHER),)
FFLAGS += -DFISHER
EXTCAMBFILES = Matrix_utils.o
else
EXTCAMBFILES =
endif
EQUATIONS=equations_ppf

include ./Makefile_main
