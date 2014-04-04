include ../../../Makefile.mine
#-include Makefile.mine


default: interface

all: interface

interface: halofit.F90 halofit_interface.F90
	$(F90) $(FFLAGS) -fPIC -c -o halofit.o halofit.F90
	$(F90) $(FFLAGS)  -fPIC -shared -o halofit_interface.so halofit_interface.F90 halofit.o $(LDFLAGS)


clean:
	rm -f *.o *~ *.flc smith2demo fsmith2  halofit_interface.so 
