include ../Makefile.mine


FFLAGS += -Iinclude -I$(CFITSIO_INC) -fPIC
CFLAGS += -Iinclude  -I$(CFITSIO_INC)
LDFLAGS += -L$(CFITSIO_LIB) -lcfitsio
CFLAGS += -fPIC

ifeq ($(strip $(PREFIX)),)
	PREFIX_FLAG=--user
elseps
	PREFIX_FLAG=--prefix=$(PREFIX)
endif

all: check names cwrapper pywrapper fwrapper c_py_bridge f_py_bridge inih

inih: inih/ini.c inih/ini.h  
	#
	# ----- COMPILING INI FILE HANDLING------
	#
	$(CC) $(CFLAGS) -I./inih -c inih/ini.c -o inih/ini.o
	ar rc lib/libini.a inih/ini.o
	ranlib lib/libini.a
	cp inih/ini.h include/


#Make the very small library from this code.
cwrapper: names C/internal_fits.c C/internal_fits.h inih
	#
	#----- COMPILING CORE C CODE ------
	#
	$(CC) $(CFLAGS) -c C/internal_fits.c -o C/internal_fits.o
	$(CC) $(CFLAGS) -c C/des_options.c -o C/des_options.o
	ar rc lib/libcfitswrap.a C/internal_fits.o C/des_options.o
	ranlib lib/libcfitswrap.a
	cp C/*.h include/

fwrapper: names cwrapper fortran/iso_c_utilities.f90 fortran/dynamic_loading.f90 fortran/f90_desglue.F90
	#
	# ----- COMPILING FORTRAN BINDINGS ------
	#
	$(F90) $(FFLAGS) -c fortran/iso_c_utilities.f90 -o fortran/iso_c_utilities.o
	$(F90) $(FFLAGS) -c fortran/des_section_names.f90 -o fortran/des_section_names.o
	$(F90) $(FFLAGS) -c fortran/dynamic_loading.f90 -o fortran/dynamic_loading.o
	$(F90) $(FFLAGS) -c fortran/f90_desglue.F90 -o fortran/f90_desglue.o
	$(F90) $(FFLAGS) -c fortran/f90_des_options.f90 -o fortran/f90_des_options.o
	ar rc lib/libffitswrap.a fortran/f90_desglue.o fortran/f90_des_options.o fortran/dynamic_loading.o  fortran/iso_c_utilities.o
	ranlib lib/libffitswrap.a
	mv *.mod include/

pywrapper: names python/pydesglue.c python/pydesglue.h python/setup.py
	#
	#----- COMPILING PYTHON MODULE------
	#
	cd python; CFITSIO_INC=$(CFITSIO_INC) CFITSIO_LIB=$(CFITSIO_LIB) $(PYTHON) setup.py install  --quiet $(PREFIX_FLAG)

c_py_bridge: C/des_python_caller.c C/des_python_caller.h
	#
	# ----- COMPILING PYTHON-C BRIDGE ------
	#
	cd C; $(CC) $(CFLAGS) `../python_compilation_flags include` -c des_python_caller.c
	ar rc lib/libcpywrap.a C/des_python_caller.o
	cp C/des_python_caller.h include/

f_py_bridge: c_py_bridge fwrapper fortran/f90_python_caller.f90
	#
	# '----- COMPILING PYTHON-FORTRAN BRIDGE ------'
	#
	$(F90) $(FFLAGS) -c fortran/f90_python_caller.f90 -o fortran/f90_python_caller.o
	ar rc lib/libfpywrap.a fortran/f90_python_caller.o
	ranlib lib/libfpywrap.a
	mv *.mod include/

names:
	$(PYTHON) bin/generate_friendly_names.py

check:
	$(MAKE) -f Makefile.check

.PHONY: check inih names

clean:
	rm -f C/*.o
	rm -f py/*.o
	rm -f fortran/*.o
	rm -f lib/lib*.a
	rm -f include/*.mod include/*.h
	rm -rf python/build
	rm -f bin/test_fglue bin/test_finterface
	rm -f python/pydesglue/section_names.py
	rm -f C/des_section_names.h
	rm -f fortran/des_section_names.f90
