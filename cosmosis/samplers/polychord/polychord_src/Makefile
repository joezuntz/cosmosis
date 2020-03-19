ifneq ($(COSMOSIS_SRC_DIR),)
include ${COSMOSIS_SRC_DIR}/config/compilers.mk
endif

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

PCOBJECTS = utils.o abort.o array_utils.o settings.o calculate.o mpi_utils.o random_utils.o chordal_sampling.o run_time_info.o clustering.o params.o priors.o read_write.o feedback.o generate.o ini.o nested_sampling.o interfaces.o 
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

libchord_mpi.so: $(PCOBJECTS_MPI)
	$(MPIFC) -shared -o  $(CURDIR)/$@ $^ $(LIBS) $(LDFLAGS)
	
libchord.so: $(PCOBJECTS) 
	$(FC) -shared -o  $(CURDIR)/$@ $^ $(LIBS) $(LDFLAGS)

clean: 
ifneq ($(MPI),)
	-rm -f $(CHORDLIBDIR)/libchord_mpi.*
	-rm -rf mpi
endif
	-rm -f $(CHORDLIBDIR)/libchord.* *.o *.mod

