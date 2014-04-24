include ${COSMOSIS_DIR}/config/compilers.mk

all: example_c_module.so

clean:
	rm -f example-c-module.so *.o 

example_c_module.so: example_c_module.c ${COSMOSIS_DIR}/cosmosis/datablock/libcosmosis.so
	$(CC) $(CFLAGS) -shared -o $@ $+
