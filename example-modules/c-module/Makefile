include ${COSMOSIS_SRC_DIR}/config/compilers.mk

all: example_c_module.so

test:

clean:
	rm -f example_c_module.so *.o
	rm -rf example_c_module.so.dSYM/

test:
	@echo "Alas, c-module has no tests"

example_c_module.so: example_c_module.c ${COSMOSIS_SRC_DIR}/cosmosis/datablock/libcosmosis.so
	$(CC) $(CFLAGS) -shared -o $@ $+
