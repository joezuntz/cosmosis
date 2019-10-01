include ${COSMOSIS_SRC_DIR}/config/compilers.mk

ifeq ("${COSMOSIS_JULIA_FLAGS}", "")
$(error To compile this module, set COSMOSIS_JULIA_FLAGS to the output of the command julia /path/to/julia-config.jl --allflags  (Might be in /usr/local/share/julia))
endif


#FLAGS=-std=gnu99 -I'/usr/local/include/julia' -DJULIA_ENABLE_THREADING=1 -fPIC -L'/usr/local/lib' -Wl,--export-dynamic -Wl,-rpath,'/usr/local/lib' -Wl,-rpath,'/usr/local/lib/julia' -ljulia -I${COSMOSIS_SRC_DIR}
FLAGS=-I${COSMOSIS_SRC_DIR} ${COSMOSIS_JULIA_FLAGS}

all: libcosmosis_julia.so

libcosmosis_julia.so: cosmosis_julia.c
	$(CC) -shared cosmosis_julia.c -o libcosmosis_julia.so ${FLAGS}

clean:
	rm -rf libcosmosis_julia.so *.o
