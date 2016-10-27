ifeq (,$(COSMOSIS_SRC_DIR))
$(error "You must source config/setup-cosmosis before building.")
endif

ifeq (1,${COSMOSIS_ALT_COMPILERS})
#Assume compilers already defined
else
CXX=g++
CC=gcc
FC=gfortran
MPIFC=mpif90
endif

ifeq (1,$(COSMOSIS_DEBUG))
COMMON_FLAGS=-O0 -g -fPIC  -fno-omit-frame-pointer
else
COMMON_FLAGS=-O3 -g -fPIC
endif

COMMON_C_FLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}
PEDANTIC_C_FLAGS=-Wall -Wextra -pedantic
CXXFLAGS=$(COMMON_C_FLAGS) $(USER_CXXFLAGS) -std=c++1y
CFLAGS=$(COMMON_C_FLAGS) $(USER_CFLAGS) -std=c99
FFLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/cosmosis/datablock $(USER_FFLAGS) -std=gnu -ffree-line-length-none
LDFLAGS=$(USER_LDFLAGS) -L${COSMOSIS_SRC_DIR}/cosmosis/datablock
PYTHON=python
MAKEFLAGS += --print-directory

ifeq (1,$(COSMOSIS_DEBUG))
LDFLAGS+=
endif

ifeq (1,${COSMOSIS_OMP})
COMMON_FLAGS+= -fopenmp
LDFLAGS+=-lgomp
endif
