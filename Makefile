include config/compilers.mk

SUBDIRS=cosmosis example-modules cosmosis-standard-library

.PHONY: all

all: 
	$(process_subdirs)

test:
	$(process_subdirs)

clean:
	$(process_subdirs)


include config/subdirs.mk

#CC=gcc
#F90=gfortran
#FFLAGS=-ffree-line-length-none -I ../../../datablock  -I
#../../../../../ -O0 -g
#LDFLAGS= -L../../../datablock -lcosmosis_fortran -lcosmosis
#CFLAGS=-DHAS_RTLD_DEFAULT -I ../../../../ -I ../../../datablock  -O0 -g
