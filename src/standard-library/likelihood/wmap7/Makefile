#Top level Makefile
include ../../Makefile.mine

#Optional (that's what the dash means) Makefile locally
-include Makefile.mine


ifeq ($(WMAP_DATA_DIR),)
ifneq ($(MAKECMDGOALS),clean)
$(error To compile the WMAP likelihood you must specify WMAP_DATA_DIR=/path/to/wmap/data in your Makefile.mine or cmb/wmap7/Makefile.mine)
endif
endif

#Makefile --- WMAP Likelihood code...


#			Non-test object files.
#
WMAPLIB = libwmap7.a
OBJS = read_archive_map.o \
	read_fits.o \
	healpix_types.o \
	br_mod_dist.o \
	WMAP_7yr_options.o \
	WMAP_7yr_util.o \
	WMAP_7yr_gibbs.o \
	WMAP_7yr_tt_pixlike.o \
	WMAP_7yr_tt_beam_ptsrc_chisq.o \
	WMAP_7yr_teeebb_pixlike.o \
	WMAP_7yr_tetbeebbeb_pixlike.o \
	WMAP_7yr_likelihood.o


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

#F90    = gfortran
#FFLAGS = -O3 $(WMAPFLAGS) -ffixed-line-length-none -ffree-line-length-none -DWMAP_DATA_DIR='"/Users/joe/src/packages/likelihood_v4p1/data/"'
#INCS   = -I. -I$(CFITSIO)/include 
#LIBS   = -L. -L$(CFITSIO)/lib -lcfitsio -framework veclib
FFLAGS+= -fPIC -I. -I$(CFITSIO_INC) -DWMAP_DATA_DIR='"$(WMAP_DATA_DIR)/"'
LDFLAGS+=-L. -L$(CFITSIO_LIB) -lcfitsio

#			Rules.
#
all: $(WMAPLIB) test interface

check: test
	./test

$(WMAPLIB): $(OBJS)
	ar r $@ $^

test: $(OBJS) test.o
	$(F90) $(FFLAGS) -o $@ test.o $(OBJS) $(LDFLAGS)

%.o: %.f90
	$(F90) $(FFLAGS) $(INCS) -c -o $@ $<

%.o: %.F90
	$(F90) $(FFLAGS) $(INCS) -c -o $@ $<

clean:
	$(RM) *.o *.mod *.log *~ *.a test *.so
	$(RM) -rf *.dSYM

distclean: clean
	$(RM) test

interface: $(WMAPLIB)
	$(F90) $(FFLAGS) wmap_interface.f90 -shared -fPIC -o wmap_interface.so -L. -lwmap7 -L../../glue/lib -lcfitswrap -lffitswrap $(INCS) $(LDFLAGS) -I../../glue/include


py: $(WMAPLIB)
	f2py -m _wmappy -c pywmap.F90 -L. -lwmap7 --link-lapack_opt -lcfitsio
