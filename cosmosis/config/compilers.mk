ifeq (,$(COSMOSIS_SRC_DIR))
$(error "You must source config/setup-cosmosis before building.")
endif

OS=$(shell uname -s)

ifeq (1,$(COSMOSIS_DEBUG))
  COMMON_FLAGS=-O0 -g -fPIC  -fno-omit-frame-pointer
else
  COMMON_FLAGS=-O3 -g -fPIC
endif

# Might be using 
ifeq (1,${COSMOSIS_OMP})
	ifneq (, $(COSMOSIS_OMP_FLAGS))
		COMMON_FLAGS+=$(COSMOSIS_OMP_FLAGS)
		LDFLAGS+=$(COSMOSIS_OMP_LDFLAGS)
	else ifeq (Darwin, $(OS))
		COSMOSIS_OMP_FLAGS=-Xpreprocessor -fopenmp -L/usr/local/lib -lomp
		COMMON_FLAGS+= -Xpreprocessor -fopenmp
		LDFLAGS+= -L/usr/local/lib -lomp
	else
		COSMOSIS_OMP_FLAGS=-fopenmp -lgomp
		COMMON_FLAGS+= -fopenmp
		LDFLAGS+=-lgomp
	endif
endif

COMMON_C_FLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/
PEDANTIC_C_FLAGS=-Wall -Wextra -pedantic
CXXFLAGS=$(COMMON_C_FLAGS) $(USER_CXXFLAGS) -std=c++14 
CFLAGS=$(COMMON_C_FLAGS) -std=c99 $(USER_CFLAGS)
FFLAGS=$(COMMON_FLAGS) -I${COSMOSIS_SRC_DIR}/datablock -std=gnu -ffree-line-length-none $(USER_FFLAGS)
#LDFLAGS=$(USER_LDFLAGS) -L${COSMOSIS_SRC_DIR}/datablock -Wl,-rpath,$(COSMOSIS_SRC_DIR)/datablock
LDFLAGS+=$(USER_LDFLAGS) -L${COSMOSIS_SRC_DIR}/datablock
PYTHON=python
MAKEFLAGS += --print-directory

ifeq (1,$(COSMOSIS_DEBUG))
LDFLAGS+=
endif

ifeq (Darwin, $(OS))
  LDFLAGS+=-headerpad_max_install_names
endif
