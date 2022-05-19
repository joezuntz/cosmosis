ifneq (clean,$(MAKECMDGOALS))
include ${COSMOSIS_SRC_DIR}/config/compilers.mk
endif

USER_CFLAGS=$(PEDANTIC_C_FLAGS) -Werror
USER_CXXFLAGS=$(PEDANTIC_C_FLAGS)

ifneq ($(CONDA_PREFIX),)
LDFLAGS+= -L$(CONDA_PREFIX)/lib 
endif

detected_OS := $(shell uname -s)
ifeq ($(detected_OS),Darwin)
RPATH=-install_name @rpath/libcosmosis.so
else
RPATH=
endif

# Find the path to libgfortran, but only if not doing make clean
ifneq (clean,$(filter clean,$(MAKECMDGOALS)))
GFORTRAN_LIB_PATH=$(shell ${FC} --print-file-name libgfortran.dylib)
ifeq (${GFORTRAN_LIB_PATH},libgfortran.dylib)
GFORTRAN_LIB_PATH=$(shell ${FC} --print-file-name libgfortran.so)
endif
endif

ifneq ($(GFORTRAN_LIB_PATH),)
LDFLAGS+=-L $(dir ${GFORTRAN_LIB_PATH})
endif
$(info ${LDFLAGS})
all: names libcosmosis.so

clean:
	rm -f *.o *.d *.so *.log *.mod *.mod 
	rm -rf  *.dSYM/


names: section_names.h cosmosis_section_names.F90 cosmosis_py/section_names.py

section_names.h: section_names.txt generate_sections.py
	python generate_sections.py section_names.txt c section_names.h

cosmosis_section_names.F90: section_names.txt generate_sections.py
	python generate_sections.py section_names.txt fortran cosmosis_section_names.F90

cosmosis_py/section_names.py: section_names.txt generate_sections.py
	python generate_sections.py section_names.txt python cosmosis_py/section_names.py


.PHONY:  clean all names


libcosmosis.so: datablock.o entry.o section.o c_datablock.o datablock_logging.o cosmosis_section_names.o cosmosis_types.o cosmosis_wrappers.o cosmosis_modules.o handler.o
	$(CXX) $(LDFLAGS) -shared $(RPATH) -o $(CURDIR)/$@ $+ -lgfortran

%.o: %.F90
	$(FC) $(FFLAGS) -c  -o $(CURDIR)/$@ $+

cosmosis_modules.o: cosmosis_types.o cosmosis_wrappers.o cosmosis_section_names.o
cosmosis_wrappers.o: cosmosis_types.o
datablock.o: section_names.h datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h datablock_logging.h datablock_types.h
c_datablock.o: section_names.h c_datablock.cc datablock.hh c_datablock.h entry.hh datablock_status.h ndarray.hh datablock_types.h
datablock_logging.o: datablock_logging.cc datablock_logging.h
entry.o: entry.cc entry.hh datablock_status.h
section.o: section.cc section.hh entry.hh datablock_status.h datablock_types.h
