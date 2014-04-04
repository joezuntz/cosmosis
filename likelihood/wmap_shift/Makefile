include ../../Makefile.mine
-include Makefile.mine

FFLAGS+=-I../../glue/include
LDFLAGS+=-L../../glue/lib -lffitswrap -lcfitswrap -lcfitsio -lini

wmap_shift.so: wmap_shift.f90
	$(F90) $(FFLAGS) -shared -o wmap_shift.so wmap_shift.f90 $(LDFLAGS) 

clean:
	rm -f *.o *.mod wmap_shift.so