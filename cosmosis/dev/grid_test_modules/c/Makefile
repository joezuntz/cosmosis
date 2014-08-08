include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_LDFLAGS= -lcosmosis

all: module.so

module.so: module.c
	$(CC) $(CFLAGS)  -shared -o module.so module.c $(LDFLAGS)

clean:
	rm -rf module.so module.so.dSYM

test:
	@echo Alas, shear/spectra has no tests

