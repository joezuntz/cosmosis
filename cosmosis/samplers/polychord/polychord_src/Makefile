include ../../../../config/compilers.mk

MPI = $(shell which $(MPIFC) 2> /dev/null)

TARGETS=libchord.so
ifneq ($(MPI),)
TARGETS+=libchord_mpi.so
endif

all: $(TARGETS)

ifeq ($(COSMOSIS_DEBUG),1)
FFLAGS += -fPIC -ffree-line-length-none
else
FFLAGS += -O3 -fPIC -ffree-line-length-none
endif

FFLAGS += -fno-stack-arrays

CHORDLIBDIR = ./

PCOBJECTS = abort.o array_utils.o calculate.o chordal_sampling.o clustering.o feedback.o generate.o ini.o interfaces.o mpi_utils.o nested_sampling.o params.o priors.o random_utils.o read_write.o run_time_info.o settings.o utils.o
PCOBJECTS_MPI = $(patsubst %.o,mpi/%.o,$(PCOBJECTS))


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
	@echo "Alas, PolyChord has no tests"

clean: 
ifneq ($(MPI),)
	-rm -f $(CHORDLIBDIR)/libchord_mpi.*
	-rm -rf mpi
endif
	-rm -f $(CHORDLIBDIR)/libchord.* *.o *.mod

