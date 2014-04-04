#Top level Makefile
#include ../../Makefile.mine

#Optional (that's what the dash means) Makefile locally
-include Makefile.mine


ifeq ($(WMAP9_DATA_DIR),)
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(MAKECMDGOALS),distclean)
$(error To compile the WMAP 9-year likelihood you must specify WMAP9_DATA_DIR=/path/to/wmap/data in your Makefile.mine or likelihood/wmap9/Makefile.mine)
endif
endif
endif

#Makefile --- WMAP Likelihood code...

#
#			Non-test object files.
#
# Need to get the right name for Mac OS X...
WMAPLIBNAME = wmap9
WMAPLIB=lib${WMAPLIBNAME}.a

OBJS = read_archive_map.o \
	read_fits.o \
	healpix_types.o \
	br_mod_dist.o \
	WMAP_9yr_options.o \
	WMAP_9yr_util.o \
	WMAP_9yr_gibbs.o \
	WMAP_9yr_tt_pixlike.o \
	WMAP_9yr_tt_beam_ptsrc_chisq.o \
	WMAP_9yr_teeebb_pixlike.o \
	WMAP_9yr_tetbeebbeb_pixlike.o \
	WMAP_9yr_likelihood.o \
	ctor.o
#
#			General Commands.
#
DIFF = diff -w
RM = rm -f

# See the CHANGES files for a description of these options
WMAPFLAGS  = -DOPTIMIZE
#WMAPFLAGS += -DUSE_LOWELL_TBEB    # turns on low-l maxlike TB/EB
#WMAPFLAGS += -DUSE_HIGHELL_TB     # turns on high-l master TB
#WMAPFLAGS += -DFASTERTT           # speed up matrix low-l TT
#WMAPFLAGS += -DTIMING             # print out timing stats for profiling

#
#			Compiler/linker configuration.  Several samples
#			are supplied.
#

## Linux GCC compiler and UPS products

F90    = gfortran
# We do not specify FFLAGS here, because we define the in the build_wmap.sh script.
#FFLAGS = -O2 -fpic $(WMAPFLAGS) 
#INCS   = -I. -I$(CFITSIO_INC)
#LIBS = -L. -L/usr/local/lib -llapack -lblas -lcfitsio

FFLAGS+= -fPIC -I.  -DWMAP9_DATA_DIR='"$(WMAP9_DATA_DIR)/"'
LDFLAGS+=-L. -lcfitsio

#
#			Rules.
#
PROGRAMS = test 

all: $(PROGRAMS)  interface


check: test
	./test

test: test.F90 $(WMAPLIB)
	$(F90) $(FFLAGS) -o $@ test.F90  -L. -l$(WMAPLIBNAME) $(LDFLAGS)

%.o: %.c
	$(CC) -c -fPIC $(CCFLAGS) -o $@ $^

$(WMAPLIB): $(OBJS)
	ar rc $@ $^

%: $(OBJS) %.o
	$(F90) $(FFLAGS) -o $@ $^

%.o: %.f90
	$(F90) -ffree-line-length-none $(FFLAGS) $(INCS) -c -o $@ $<

%.o: %.F90
	$(F90) -ffree-line-length-none $(FFLAGS) $(INCS) -c -o $@ $<

clean:
	$(RM) -r *.o *.mod *.log *~ *.a *.so *.dSYM

distclean: clean
	$(RM) $(WMAPLIB) $(PROGRAMS)

interface: $(WMAPLIB)
	$(F90) $(FFLAGS) wmap_interface.f90 -shared -fPIC -o wmap_interface.so  -L. -l${WMAPLIBNAME}  $(INCS) $(LDFLAGS) -lcosmosis_fortran -lcfitsio




