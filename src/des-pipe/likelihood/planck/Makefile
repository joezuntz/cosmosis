include ../../Makefile.mine
-include Makefile.mine


#CC = gcc
#FC = ifort
FC = $(F90)

SO=so

VERSION=1.0-PACKAGED-WITH-COSMOSIS

#DEFINESMACOS = -D HAS_RTLD_DEFAULT
#DEFINES = -D HAS_RTLD_DEFAULT -D HAS_LAPACK -D LAPACK_CLIK -D NOHEALPIX -D CLIK_LENSING -D CLIKSVNVERSION='"$(VERSION)"'
DEFINES =  -D HAS_LAPACK -D LAPACK_CLIK -D NOHEALPIX -D CLIK_LENSING -D CLIKSVNVERSION='"$(VERSION)"'

COLORS = 1

SOURCE_DIR=clik


#temporary dirs
BDIR := ./lib
ODIR := ./build

# tools
LD = gcc
INSTALL = install
ECHO = echo

# get the os
UNAME := $(shell uname -s)
FRUNTIME = $(FFLAGS)
SHARED = -shared

FMODULEPATH?=-J

# get version of the code from the svn version




# final CFLAG and FFLAGS
CFLAGS+=$(DEFINES) -I $(SOURCE_DIR) -I $(SOURCE_DIR)/cldf -I $(SOURCE_DIR)/minipmc -I $(SOURCE_DIR)/lenslike/plenslike
FFLAGS+=$(FM64) $(FOPENMP) $(FFPIC) $(DEFINES)  $(FMODULEPATH) $(ODIR)

# Lapack section
CFLAGS+=-I../../glue/include 
LDFLAGS+=-L../../glue/lib -lcfitswrap -lini -lcfitsio
LDFLAG=$(LDFLAGS)
#LDFLAG = $(CM64) $(CFITSIO) $(LAPACK) $(FRUNTIME) -ldl -lm -lpthread

# define some path to find the codes
vpath %.c $(SOURCE_DIR) $(SOURCE_DIR)/minipmc $(SOURCE_DIR)/cldf $(SOURCE_DIR)/CAMspec $(SOURCE_DIR)/component_plugin/basic $(SOURCE_DIR)/lenslike/plenslike
vpath %.f90  $(SOURCE_DIR) $(SOURCE_DIR)/minipmc $(SOURCE_DIR)/cldf $(SOURCE_DIR)/CAMspec $(SOURCE_DIR)/gibbs $(SOURCE_DIR)/act_spt $(SOURCE_DIR)/lowlike
vpath  %.F90 $(SOURCE_DIR) $(SOURCE_DIR)/minipmc $(SOURCE_DIR)/cldf $(SOURCE_DIR)/CAMspec $(SOURCE_DIR)/gibbs $(SOURCE_DIR)/act_spt $(SOURCE_DIR)/lowlike

# define color output if needed
ifeq ($(COLORS),1)
NO_COLOR=\x1b[0m
GREEN_COLOR=\x1b[32;11m
RED_COLOR=\x1b[31;01m
BLUE_COLOR=\x1b[35;11m
endif

ifeq ($(shell uname),Darwin)
	RPATH_FLAGS=-Xlinker -rpath -Xlinker $(PWD)
else
	RPATH_FLAGS=-Wl,-rpath,$(PWD)
endif

# all the code
TOOLS := $(addprefix $(ODIR)/,errorlist.o io.o distribution.o cldf.o)
CLIKMAIN := $(addprefix $(ODIR)/,clik.o lklbs.o lowly_common.o clik_helper.o)
CLIKLKL := $(addprefix $(ODIR)/,clik_lowlike.o clik_actspt.o clik_gibbs.o clik_CAMspec.o)
LENSLKL := $(addprefix $(ODIR)/,plenslike_dat_mono.o plenslike_dat_quad.o qest.o wignerd.o)
ACTSPTLKL := $(addprefix $(ODIR)/,Highell_options.f90.o Highell_subroutines.f90.o  Foregrounds_loading.f90.o ACT_equa_likelihood.f90.o SPT_reichardt_likelihood.f90.o ACT_south_likelihood.f90.o  SPT_keisler_likelihood.f90.o  Highell_likelihood.f90.o clik_actspt.f90.o)
CAMSPECLKL := $(addprefix $(ODIR)/,CAMspec.f90.o clik_CAMspec.f90.o)
LOWLIKELKL := $(addprefix $(ODIR)/,healpix_types.f90.o read_archive_map.f90.o read_fits.f90.o br_mod_dist.f90.o Planck_options.f90.o  Planck_teeebb_pixlike.f90.o  Planck_likelihood.f90.o clik_lowlike.f90.o)
GIBBSLKL := $(addprefix $(ODIR)/,comm_br_mod.f90.o clik_gibbs.f90.o)
CLIKLKL_F90:= $(ACTSPTLKL) $(CAMSPECLKL) $(GIBBSLKL) $(LOWLIKELKL)

CLIKLIB := $(TOOLS) $(CLIKMAIN) $(CLIKLKL) $(CLIKLKL_F90) $(LENSLKL) $(LAPACKDEP)


all: libclik.$(SO) tidy planck_interface.so



#$(BDIR)/libclik_f90.$(SO) tidy



install_dir: 
	@mkdir -p $(PREFIX)/bin
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	@mkdir -p $(PREFIX)/share/clik

install: $(BDIR)/libclik.$(SO) $(BDIR)/libclik_f90.$(SO) $(BDIR)/clik_example_C $(BDIR)/clik_example_f90 $(LAPACKDEP) $(BDIR)/clik_profile.sh $(BDIR)/clik_profile.csh $(BDIR)/clik-config $(BDIR)/clik-config_f90 | install_dir
	@$(ECHO) "install libs $(BLUE_COLOR)libclik.$(SO) libclik_f90.$(SO)$(NO_COLOR) in $(BLUE_COLOR)$(PREFIX)/lib $(NO_COLOR)"
	@$(INSTALL)  $(BDIR)/libclik.$(SO) $(BDIR)/libclik_f90.$(SO) $(LAPACKDEP) $(PREFIX)/lib
	@$(ECHO) "install includes $(BLUE_COLOR)clik.h clik.mod$(NO_COLOR) in $(BLUE_COLOR)$(PREFIX)/include $(NO_COLOR)"
	@$(INSTALL)  $(SOURCE_DIR)/clik.h $(SOURCE_DIR)/minipmc/maths_base.h $(SOURCE_DIR)/minipmc/errorlist.h $(SOURCE_DIR)/minipmc/io.h $(SOURCE_DIR)/lapack_clik.h $(SOURCE_DIR)/minipmc/pmc.h $(ODIR)/clik.mod $(PREFIX)/include
	@$(ECHO) "install clik_profile & clik-config$(BLUE_COLOR)clik_profile.sh clik_profile.csh clik-config clik-config_f90$(NO_COLOR) in $(BLUE_COLOR)$(PREFIX)/bin $(NO_COLOR)"
	@$(INSTALL)  $(BDIR)/clik_profile.sh $(BDIR)/clik_profile.csh $(BDIR)/clik-config $(BDIR)/clik-config_f90 $(PREFIX)/bin
	@$(ECHO) "install exec tools $(BLUE_COLOR)clik_example_C clik_example_f90$(NO_COLOR) in $(BLUE_COLOR)$(PREFIX)/bin $(NO_COLOR)"
	@$(INSTALL)  $(BDIR)/clik_example_C $(BDIR)/clik_example_f90 $(PREFIX)/bin


ifdef PYTHON
PYTHONPATH = $(PREFIX)/lib/`$(PYTHON) -c"import sys;print 'python%s/site-packages'%sys.version[0:3]"`
PYTHONEXE := `which $(PYTHON)`
else
PYTHONPATH := 
endif

$(BDIR)/clik_profile.sh: $(SOURCE_DIR)/clik_profile.sh.template |$(BDIR)
	@sed "s!PREFIX!$(PREFIX)!g;s/DYLD_LIBRARY_PATH/$(LIBPATHNAME)/g;s@CFITSIOLIBPATH@$(CFITSIO_LIBPATH)@g;s!FORTRANLIBPATH!$(FLIBPATH)!g;s!LAPACKLIBPATH!$(LAPACKLIBPATH)!g;s!MPYTHONPATH!$(PYTHONPATH)!g" <$< >$@

$(BDIR)/clik_profile.csh: $(SOURCE_DIR)/clik_profile.csh.template |$(BDIR)
	@sed "s!PREFIX!$(PREFIX)!g;s/DYLD_LIBRARY_PATH/$(LIBPATHNAME)/g;s@CFITSIOLIBPATH@$(CFITSIO_LIBPATH)@g;s!FORTRANLIBPATH!$(FLIBPATH)!g;s!LAPACKLIBPATH!$(LAPACKLIBPATH)!g;s!MPYTHONPATH!$(PYTHONPATH)!g" <$< >$@

$(BDIR):
	@mkdir $(BDIR)

#| $(BDIR)
$(ODIR): 
	@mkdir $(ODIR)

$(CLIKLIB): | $(ODIR) $(ODIR)/.print_info 

planck_interface.so: libclik.$(SO) planck_interface.c
	$(CC) $(CFLAGS) $(CFLAGS) -L$(PWD) -lclik $(RPATH_FLAGS) $(SHARED) -o $@ planck_interface.c  $(LDFLAGS) 
	# $(CC) $(CFLAGS) $(CFLAGS) -L$(PWD)  -lclik $(SHARED) -o $@ planck_interface.c  $(LDFLAGS) 


#$(ODIR)/.test_cfitsio

libclik.$(SO): $(CLIKLIB) 
	@$(ECHO) "build $(BLUE_COLOR)$(@) $(NO_COLOR)"
	$(LD)  $(SHARED)  $(LAPACK) $(LDFLAG) $(RPATH_FLAGS)  $^ -o $@
#	$(LD)  $(SHARED)  $(LAPACK) $(LDFLAG)  $^ -o $@

$(BDIR)/libclik_f90.$(SO): $(BDIR)/libclik.$(SO) $(addprefix $(ODIR)/,clik_fortran.o clik.f90.o)
	@$(ECHO) "build $(BLUE_COLOR)$(@) $(NO_COLOR)"
	$(LD) $(SHARED)  $(LDFLAG) $(LAPACK) -L$(BDIR) -lclik $^ -o $@

$(BDIR)/clik_example_C: $(ODIR)/clik_example_c.o $(BDIR)/libclik.$(SO)
	@$(ECHO) "build $(BLUE_COLOR)$(@) $(NO_COLOR)"
	$(CC) $(LDFLAG) $(LAPACK) -L$(BDIR) -lclik $< -o $@

$(BDIR)/clik_example_f90: $(ODIR)/clik_example_f90.f90.o $(BDIR)/libclik_f90.$(SO)
	@$(ECHO) "build $(BLUE_COLOR)$(@) $(NO_COLOR)"
	$(FC) $(LDFLAG) $(LAPACK)  -L$(BDIR) -lclik_f90 -lclik $< -o $@

$(BDIR)/liblapack_clik.$(SO): |$(BDIR)
ifndef MKL_LIB_FULLPATH
	@$(ECHO) "$(RED_COLOR)I suspect an error with your MKLROOT, or MKL_LIB_FULLPATH, please check$(NO_COLOR)"
endif
	@$(ECHO) "build $(BLUE_COLOR)$(@) $(NO_COLOR),"
	@$(ECHO) "(see chapter 5 in http://software.intel.com/sites/products/documentation/hpc/mkl/lin/)"
	@$(ECHO) "using the following command line:"
	gcc $(SHARED)  $(MKL_TO_INCLUDE) -Wl,--start-group $(MKL_LIB_FULLPATH) -Wl,--end-group -L$(IFORTLIBPATH) -L/lib -L/lib64 -liomp5 -lpthread -lm -o $@

$(ODIR)/%.o : %.c 
	@$(ECHO) "$(GREEN_COLOR)$< $(NO_COLOR) -> $(GREEN_COLOR) $(@) $(NO_COLOR)"
	$(CC) -c $(CFLAGS) $< -o$(@)

$(ODIR)/%.f90.o : %.f90 
	@$(ECHO) "$(GREEN_COLOR)$< $(NO_COLOR) -> $(GREEN_COLOR) $(@) $(NO_COLOR)"
	$(FC) -c $(FFLAGS) $< -o$(@)

$(ODIR)/%.f90.o : %.F90 
	@$(ECHO) "$(GREEN_COLOR)$< $(NO_COLOR) -> $(GREEN_COLOR) $(@) $(NO_COLOR)"
	$(FC) -c $(FFLAGS) $< -o$(@)

$(ODIR)/%.py: $(SOURCE_DIR)/python/%.py
	@sed "s@PYTHONEXE@$(PYTHONEXE)@g;s@REPLACEPATH@$(PYTHONPATH)@g" <$< >$@
	@$(INSTALL) $@ $(PREFIX)/bin/$(subst .py,,$(@F))

$(ODIR)/.print_info: |$(ODIR)
	@$(ECHO) "\n$(BLUE_COLOR)Compile$(NO_COLOR) clik $(VERSION) "
	@$(ECHO) "$(BLUE_COLOR)Using $(NO_COLOR) CC = $(CC)"
	@$(ECHO) "$(BLUE_COLOR)Using $(NO_COLOR) FC = $(FC)"
	@$(ECHO) "$(BLUE_COLOR)Using $(NO_COLOR) CFLAGS = $(CFLAGS)"
	@$(ECHO) "$(BLUE_COLOR)Using $(NO_COLOR) FFLAGS = $(FFLAGS)"
	@$(ECHO) "$(BLUE_COLOR)Using the following lapack link line:$(NO_COLOR) $(LAPACK)"
	@$(ECHO) "$(BLUE_COLOR)Using the following cfitsio link line:$(NO_COLOR) $(CFITSIO)"
	@$(ECHO) "$(BLUE_COLOR)Using the following fortran runtime link line:$(NO_COLOR) $(FRUNTIME)"
	@$(ECHO) "$(BLUE_COLOR)Build dir:$(NO_COLOR) $(BDIR)"
	@$(ECHO)
	@touch $(@)

install_python: install $(addprefix $(ODIR)/, clik_add_free_calib.py clik_explore_1d.py prepare_actspt.py clik_get_selfcheck.py clik_example_py.py clik_join.py clik_disjoin.py clik_print.py prepare_wmap.py clik_extract_external.py) |$(ODIR)
	@LINK_CLIK="$(LDFLAG) $(LAPACK) -L$(PREFIX)/lib -lclik " $(PYTHON) setup.py build --build-base=$(ODIR) install --install-lib=$(PYTHONPATH)

HAS_CFITSIO_INC := $(shell [ -f $(CFITSIO_INCPATH)/fitsio.h ] && echo OK)
HAS_CFITSIO_LIB := $(shell [ -f $(CFITSIO_LIBPATH)/libcfitsio.$(SO) ] && echo OK)

$(ODIR)/.test_cfitsio: |$(ODIR)
ifneq ($(HAS_CFITSIO_INC),OK)
	@$(ECHO) "\n$(RED_COLOR)Cannot find cfisio includes ($(CFITSIO_INCPATH)/fitsio.h)$(NO_COLOR)"
	@false
endif
ifneq ($(HAS_CFITSIO_LIB),OK)
	@$(ECHO) "\n$(RED_COLOR)Cannot find cfisio lib ($(CFITSIO_LIBPATH)/libcfitsio.$(SO))$(NO_COLOR)"
	@false
endif
	@touch $(@)


INSTALL_CFLAG = $(subst ",\",$(subst ',\',$(CM64) $(COPENMP) $(CFPIC) $(DEFINES) -I $(PREFIX)/include $(INCLUDES)))
INSTALL_FFLAG = $(subst ",\",$(subst ',\',$(FM64) $(FOPENMP) $(FFPIC) $(DEFINES) $(FMODULEPATH) $(PREFIX)/include))
INSTALL_CLIB = $(CM64) $(CFITSIO) $(FRUNTIME) -ldl -lm -lpthread $(CFITSIO) $(FRUNTIME) 
INSTALL_FLIB = $(CM64) $(CFITSIO) $(FRUNTIME) -ldl -lm -lpthread $(CFITSIO) $(FRUNTIME) 
ifdef LAPACK_INSTALL
INSTALL_CLIB += $(LAPACK_INSTALL)
INSTALL_FLIB += $(LAPACK_INSTALL)
else
INSTALL_CLIB += $(LAPACK)
INSTALL_FLIB += $(LAPACK)
endif
INSTALL_CLIB += -L$(PREFIX)/lib -lclik
INSTALL_FLIB += -L$(PREFIX)/lib -lclik -lclik_f90

$(BDIR)/clik-config: $(SOURCE_DIR)/clik-config.template |$(BDIR)
	@sed "s@CFLAG@$(INSTALL_CFLAG)@g;s@LIB@$(INSTALL_CLIB)@g" <$< >$@

$(BDIR)/clik-config_f90: $(SOURCE_DIR)/clik-config.template |$(BDIR)
	@sed "s@CFLAG@$(INSTALL_FFLAG)@g;s@LIB@$(INSTALL_FLIB)@g" <$< >$@


clean:
	@$(ECHO) "$(BLUE_COLOR)Removing all $(NO_COLOR)"
	@rm -rf $(BDIR) $(ODIR)  planck_interface.so* libclik.$(SO)


.PHONY :clean  LAPACK_PRINT LAPACK_DEP tidy
