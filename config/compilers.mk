CXX=g++
CC=gcc
FC=gfortran

COMMON_C_FLAGS=-O3 -g -fPIC -Wall -Wextra -pedantic -I$(SRC_AREA) -I$(BUILD_TOP) -I$(COSMOSIS_DIR)/src $(USER_CXXFLAGS)
CXXFLAGS=$(COMMON_C_FLAGS) -std=c++11
CFLAGS=$(COMMON_C_FLAGS) -std=c99
FFLAGS=$(COMMON_C_FLAGS) -std=f2003 -fimplicit-none  -ffixed-line-length-none
LDFLAGS=$(USER_LDFLAGS) -L.

PYTHON=python
