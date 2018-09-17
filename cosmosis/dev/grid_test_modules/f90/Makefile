include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_LDFLAGS= -lcosmosis_fortran -lcosmosis

all: module_f.so

module_f.so: module_f.f90
	$(FC) $(FFLAGS)  -shared -o module_f.so module_f.f90 $(LDFLAGS)

clean:
	rm -rf module_f.so module_f.so.dSYM

test:
	@echo Alas, f90 test has no tests.  Ironic.

