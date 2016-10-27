include ../../../../config/compilers.mk

MPI = $(shell which $(MPIFC) 2> /dev/null)

TARGETS=libnest3.so
ifneq ($(MPI),)
TARGETS+=libnest3_mpi.so
endif

all: $(TARGETS)

FFLAGS += -O3 -fPIC -ffree-line-length-none

ifeq ($(LAPACK_LINK),)
LAPACKLIB = -llapack
else
LAPACKLIB = ${LAPACK_LINK}
endif

NESTLIBDIR = ./

NSOBJECTS = utils.o utils1.o priors.o kmeans_clstr.o xmeans_clstr.o posterior.o nested.o cwrapper.o
NSOBJECTS_MPI = $(patsubst %.o,mpi/%.o,$(NSOBJECTS))



%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $^ 

%.o: %.F90
	$(FC) $(FFLAGS) -c -o $@ $^ 

mpi/%.o: %.f90 mpi/.phony
	$(MPIFC) $(FFLAGS) -DMPI -c -o $@ $<

mpi/%.o: %.F90 mpi/.phony
	$(MPIFC) $(FFLAGS) -DMPI -c -o $@ $<

mpi/.phony:
	@mkdir -p mpi
	@touch mpi/.phony

test:
	@echo "Alas, MultiNest has no tests"

libnest3_mpi.so: $(NSOBJECTS_MPI)
	$(MPIFC) -shared -o  $@ $^ $(LIBS) $(LAPACKLIB) $(LDFLAGS)
	
libnest3.so: $(NSOBJECTS) 
	$(FC) -shared -o  $@ $^ $(LIBS) $(LAPACKLIB)  $(LDFLAGS)

obj_detect:
	make -C example_obj_detect

gaussian:
	make -C example_gaussian

rosenbrock:
	make -C example_rosenbrock

ackley:
	make -C example_ackley

himmelblau:
	make -C example_himmelblau

eggbox:
	make -C example_eggbox

gauss_shell:
	make -C example_gauss_shell

eggboxC:
	make -C example_eggbox_C

eggboxC++:
	make -C example_eggbox_C++

clean: 
ifneq ($(MPI),)
	-rm -f $(NESTLIBDIR)/libnest3_mpi.*
	-rm -rf mpi
endif
	-rm -f $(NESTLIBDIR)/libnest3.* *.o *.mod

cleanall: clean_exec clean clean_obj_detect clean_gaussian clean_gauss_shell clean_eggbox \
clean_example_eggbox_C clean_example_eggbox_C++ clean_rosenbrock clean_himmelblau \
clean_ackley

clean_exec:
	-rm obj_detect gaussian rosenbrock ackley himmelblau gauss_shell eggbox eggboxC eggboxC++

clean_obj_detect:
	make -C example_obj_detect clean

clean_gaussian:
	make -C example_gaussian clean

clean_rosenbrock:
	make -C example_rosenbrock clean

clean_ackley:
	make -C example_ackley clean

clean_himmelblau:
	make -C example_himmelblau clean

clean_eggbox:
	make -C example_eggbox clean

clean_gauss_shell:
	make -C example_gauss_shell clean

clean_example_eggbox_C:
	make -C example_eggbox_C clean

clean_example_eggbox_C++:
	make -C example_eggbox_C++ clean
