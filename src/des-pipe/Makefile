include Makefile.mine

ifeq ($(CFITSIO_INC),)
ifneq ($(MAKECMDGOALS),clean)
$(error You must specify the CFITSIO_INC directory in your Makefile.mine )
endif
endif


ifeq ($(CFITSIO_LIB),)
ifneq ($(MAKECMDGOALS),clean)
$(error You must specify the CFITSIO_INC directory in your Makefile.mine )
endif
endif

CFLAGS+=-I$(CFITSIO_INC) 
LDFLAGS+=-L$(CFITSIO_LIB)


TARGETS?="glue/dummy"

default: glue $(TARGETS) success


$(TARGETS): force
	@cd $@; echo '-----------------------------------------'; echo '--------- COMPILING $@' ------; echo '---------------------------------------'; $(MAKE) --print-directory

clean:
	-cd glue; $(MAKE) clean
	-for d in $(TARGETS); do (cd $$d; $(MAKE) --print-directory clean ); done
	-cd interface_example && $(MAKE) clean
	@echo Make clean complete

glue: force
	cd glue; $(MAKE)

example: glue
	cd example && $(MAKE)

force:
	true

success:
	@echo "#################################"
	@echo "## COMPILATION SUCCEEDED!      ##"
	@echo "#################################"
