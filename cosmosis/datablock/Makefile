include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_CFLAGS=$(PEDANTIC_C_FLAGS)
USER_CXXFLAGS=$(PEDANTIC_C_FLAGS)

#CXXFLAGS=-std=c++11 -O0 -Wall -Wextra -pedantic -fPIC -g
#CXXFLAGS=-std=c++11 -O0 -Wall -Wextra -pedantic -fPIC -g
#CCFLAGS=-std=c99 -O0 -Wall -Wextra -pedantic -fPIC -g
#CCFLAGS=-std=c99 -O0 -Wall -Wextra -pedantic -fPIC -g
#F90FLAGS = -Wall -fPIC -ffree-line-length-none

#CXXFLAGS+=-I../../
#CFLAGS+=-I../../

ifdef MEMCHECK
  MEMCHECK_CMD=valgrind --error-exitcode=1 --leak-check=yes \
               --errors-for-leak-kinds=definite --track-origins=yes \
               --suppressions=cosmosis_tests.supp
else
  MEMCHECK_CMD=
endif

all: libcosmosis.so libcosmosis_fortran.so c_datablock_t datablock_t section_t entry_t fortran_t cosmosis_py/section_names.py

lib: libcosmosis.so cosmosis_py/section_names.py

clean:
	rm -f *.o *.d *.so *.log *.mod cosmosis_f90/*.mod cosmosis_f90/*.o
	rm -rf  libcosmosis_fortran.so.dSYM/*

verify_setup:
	@if [ -z "$(SETUP_UPS)" ]; then echo "You must setup UPS first" && exit 1; fi

test:  test_entry test_section test_datablock test_c_datablock \
	test_c_datablock_int_array test_c_datablock_double_array test_c_datablock_complex_array \
	test_fortran

test_datablock: lib datablock_t
	@echo -n "Running datablock_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./datablock_t > datablock_t.log
	@echo  ... passed

test_c_datablock: lib c_datablock_t
	@echo -n "Running c_datablock_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./c_datablock_t > c_datablock_t.log
	@echo  ... passed

test_c_datablock_int_array: lib c_datablock_int_array_t
	@echo -n "Running c_datablock_int_array_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./c_datablock_int_array_t > c_datablock_int_array_t.log
	@echo  ... passed

test_c_datablock_double_array: lib c_datablock_double_array_t
	@echo -n "Running c_datablock_double_array_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./c_datablock_double_array_t > c_datablock_double_array_t.log
	@echo  ... passed

test_c_datablock_complex_array: lib c_datablock_complex_array_t
	@echo -n "Running c_datablock_complex_array_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./c_datablock_complex_array_t > c_datablock_complex_array_t.log
	@echo  ... passed

test_section: lib section_t
	@echo -n "Running section_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./section_t >  section_t.log
	@echo  ... passed

test_entry: lib entry_t
	@echo -n "Running entry_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./entry_t >  entry_t.log
	@echo  ... passed

test_fortran: fortran_t
	@echo -n "Running fortran_t "
	@LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} $(MEMCHECK_CMD) ./fortran_t >  fortran_t.log
	@echo  ... passed

test_python: 
	@echo -n "Running python tests - needs nose package "
	@nosetests cosmosis_py &> python_t.log
	@echo ... passed

section_names.h: section_list.txt generate_sections.py
	python generate_sections.py section_list.txt c section_names.h

cosmosis_f90/cosmosis_section_names.F90: section_list.txt generate_sections.py
	python generate_sections.py section_list.txt fortran cosmosis_f90/cosmosis_section_names.F90

cosmosis_py/section_names.py: section_list.txt generate_sections.py 
	python generate_sections.py section_list.txt python cosmosis_py/section_names.py

.PHONY: test_datablock test_c_datablock test_section test_entry test_fortran test clean test_python 

# This rule will make us recompile the Fortran source code that goes into
# libcomsosis_fortran.so every time that libcosmosis.so changes, when really
# we should only have to relink libcomsosis_fortran.so from the
# already-generated object files. However, this rule also does away with the
# problem of generating dependencies for the Fortran modules within
# libcomsosis_fortran.so. The tradeoff is worthwhile, unless so many source
# files are added that the build time is excessive.
libcosmosis_fortran.so: cosmosis_f90/cosmosis_section_names.F90 cosmosis_f90/cosmosis_types.F90 cosmosis_f90/cosmosis_wrappers.F90 cosmosis_f90/cosmosis_modules.F90 libcosmosis.so
	$(FC) $(FFLAGS) -shared -o $@ $+

fortran_t: libcosmosis_fortran.so  cosmosis_f90/cosmosis_test.F90
	$(FC) $(FFLAGS)  -o $@ cosmosis_f90/cosmosis_test.F90  -L. -lcosmosis_fortran -lcosmosis

libcosmosis.so: datablock.o entry.o section.o c_datablock.o datablock_logging.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $+

datablock_t: datablock_test.cc datablock.hh libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ datablock_test.cc -L . -lcosmosis

c_datablock_t: c_datablock_test.c  test_c_datablock_scalars.h c_datablock.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ c_datablock_test.c -L . -lcosmosis

c_datablock_int_array_t: c_datablock_int_array_test.c c_datablock.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_double_array_t: c_datablock_double_array_test.c c_datablock.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

c_datablock_complex_array_t: c_datablock_complex_array_test.c c_datablock.h libcosmosis.so
	$(CC) $(CFLAGS) -o $@ $< -L . -lcosmosis

entry_t: entry_test.o libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ $< -L . -lcosmosis

section_t: section_test.o libcosmosis.so
	$(CXX) $(CXXFLAGS) -o $@ $< -L . -lcosmosis

datablock.o: section_names.h datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h datablock_logging.h
c_datablock.o: section_names.h c_datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h
datablock_logging.o: datablock_logging.cc datablock_logging.h
entry.o: entry.cc entry.hh datablock_status.h
entry_t.o: entry_t.cc entry.hh
section.o: section.cc section.hh entry.hh datablock_status.h
section_t.o: section_t.cc section.hh



