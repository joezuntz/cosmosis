include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_CFLAGS=$(PEDANTIC_C_FLAGS) -Werror
USER_CXXFLAGS=$(PEDANTIC_C_FLAGS)

detected_OS := $(shell uname -s)
ifeq ($(detected_OS),Darwin)
RPATH=-install_name @rpath/libcosmosis.so
else
RPATH=
endif
#CXXFLAGS=-std=c++11 -O0 -Wall -Wextra -pedantic -fPIC -g
#CXXFLAGS=-std=c++11 -O0 -Wall -Wextra -pedantic -fPIC -g
#CCFLAGS=-std=c99 -O0 -Wall -Wextra -pedantic -fPIC -g
#CCFLAGS=-std=c99 -O0 -Wall -Wextra -pedantic -fPIC -g
#F90FLAGS = -Wall -fPIC -ffree-line-length-none

#CXXFLAGS+=-I../../
#CFLAGS+=-I../../

all: lib

lib: libcosmosis.so  cosmosis_py/section_names.py

clean:
	rm -f *.o *.d *.so *.log *.mod *.mod 
	rm -f c_datablock_complex_array_t c_datablock_double_array_t c_datablock_int_array_t
	rm -f c_datablock_t
	rm -f c_datablock_multidim_double_array_t
	rm -f c_datablock_multidim_int_array_t
	rm -f c_datablock_multidim_complex_array_t
	rm -f datablock_t entry_t fortran_t ndarray_t section_t
	rm -rf  *.dSYM/
	rm -f cosmosis_py/section_names.py section_names.h cosmosis_section_names.F90

verify_setup:
	@if [ -z "$(SETUP_UPS)" ]; then echo "You must setup UPS first" && exit 1; fi

test:  test_entry test_section test_datablock test_c_datablock \
	test_ndarray \
	test_c_datablock_int_array test_c_datablock_double_array test_c_datablock_complex_array \
	test_c_datablock_multidim_double_array \
	test_c_datablock_multidim_int_array \
	test_c_datablock_multidim_complex_array \
	test_fortran

test_ndarray: ndarray_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_datablock: datablock_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock: c_datablock_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_int_array: c_datablock_int_array_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_double_array:  c_datablock_double_array_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_complex_array: c_datablock_complex_array_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_multidim_double_array: c_datablock_multidim_double_array_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_multidim_int_array: c_datablock_multidim_int_array_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_c_datablock_multidim_complex_array: c_datablock_multidim_complex_array_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_section: section_t lib
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

# entry_t doesn't depend on libcosmosis
test_entry: entry_t
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_fortran: fortran_t libcosmosis.so
	@/bin/echo -n "Running $< "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./$< > $<.log
	@/bin/echo  ... passed

test_python: 
	@/bin/echo -n "Running python tests - needs nose package "
	@nosetests cosmosis_py &> python_t.log
	@/bin/echo ... passed

section_names.h: section_names.txt generate_sections.py
	python generate_sections.py section_names.txt c section_names.h

cosmosis_section_names.F90: section_names.txt generate_sections.py
	python generate_sections.py section_names.txt fortran cosmosis_section_names.F90

cosmosis_py/section_names.py: section_names.txt generate_sections.py 
	python generate_sections.py section_names.txt python cosmosis_py/section_names.py
	cp cosmosis_py/section_names.py ../names.py

.PHONY: test_datablock test_c_datablock test_section test_entry test_fortran test clean test_python 

# This rule will make us recompile the Fortran source code that goes into
# libcomsosis_fortran.so every time that libcosmosis.so changes, when really
# we should only have to relink libcomsosis_fortran.so from the
# already-generated object files. However, this rule also does away with the
# problem of generating dependencies for the Fortran modules within
# libcomsosis_fortran.so. The tradeoff is worthwhile, unless so many source
# files are added that the build time is excessive.
libcosmosis_fortran.so: cosmosis_section_names.F90 cosmosis_types.F90 cosmosis_wrappers.F90 cosmosis_modules.F90 libcosmosis.so
	$(FC) $(FFLAGS) -shared -o $(CURDIR)/$@ $+

libcosmosis.so: datablock.o entry.o section.o c_datablock.o datablock_logging.o
	$(CXX) $(CXXFLAGS) -shared -o $(CURDIR)/$@ $+



fortran_t: cosmosis_test.F90
	$(FC) $(FFLAGS)  -o $@ cosmosis_test.F90  -L. -lcosmosis_fortran -lcosmosis


datablock_t: datablock_test.cc datablock.hh libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ datablock_test.cc -L . -lcosmosis

c_datablock_t: c_datablock_test.c  test_c_datablock_scalars.h c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ c_datablock_test.c -L . -lcosmosis

c_datablock_int_array_t: c_datablock_int_array_test.c c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_double_array_t: c_datablock_double_array_test.c c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_complex_array_t: c_datablock_complex_array_test.c c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_multidim_double_array_t: c_datablock_multidim_double_array_test.c c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_multidim_int_array_t: c_datablock_multidim_int_array_test.c c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_multidim_complex_array_t: c_datablock_multidim_complex_array_test.c c_datablock.h datablock_types.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

entry_t: entry_test.o entry.o
	$(CXX) $(CXXFLAGS) -o $@ $< entry.o

section_t: section_test.o section.o entry.o
	$(CXX) $(CXXFLAGS) -o $@ $^

# Note: ndarray_t does not make use of libcosmosis!
ndarray_t: ndarray_test.o
	$(CXX) $(CXXFLAGS) -o $@ $<

datablock.o: section_names.h datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h datablock_logging.h datablock_types.h
c_datablock.o: section_names.h c_datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h ndarray.hh datablock_types.h
datablock_logging.o: datablock_logging.cc datablock_logging.h
entry.o: entry.cc entry.hh datablock_status.h
entry_test.o: entry_test.cc entry.hh
section.o: section.cc section.hh entry.hh datablock_status.h datablock_types.h
section_test.o: section_test.cc section.hh datablock_types.h
ndarray_test.o: ndarray_test.cc ndarray.hh
