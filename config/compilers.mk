CXX=g++
CC=gcc
FC=gfortran

COMMON_FLAGS=-O3 -g -fPIC -Wall -I$(SRC_AREA) -I$(BUILD_TOP) -I$(COSMOSIS_DIR)src $(USER_CXXFLAGS)
COMMON_C_FLAGS=$(COMMON_FLAGS) -Wextra -pedantic
CXXFLAGS=$(COMMON_C_FLAGS) -std=c++11
CFLAGS=$(COMMON_C_FLAGS) -std=c99
FFLAGS=$(COMMON_FLAGS) -std=gnu -fimplicit-none  -ffree-line-length-none
LDFLAGS=$(USER_LDFLAGS) -L$(BUILD_TOP)/lib -L.

PYTHON=python
