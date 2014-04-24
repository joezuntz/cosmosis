CXX=g++
CC=gcc
FC=gfortran

COMMON_FLAGS=-O3 -g -fPIC -I${COSMOSIS_DIR}
COMMON_C_FLAGS=$(COMMON_FLAGS) -Wall -Wextra -pedantic
CXXFLAGS=$(COMMON_C_FLAGS) $(USER_CXXFLAGS) -std=c++11
CFLAGS=$(COMMON_C_FLAGS) $(USER_CFLAGS) -std=c99
FFLAGS=$(COMMON_FLAGS) $(USER_FFLAGS) -std=gnu -fimplicit-none  -ffree-line-length-none
LDFLAGS=$(USER_LDFLAGS) -L${COSMOSIS}/cosmosis/datablock

PYTHON=python
