include ${COSMOSIS_SRC_DIR}/config/compilers.mk

define newline


endef



ifeq (${MINUIT2_LIB},)
    $(warning)
    $(warning ${newline} ${newline}Not building the minuit sampler as the you need the Minuit2 library ${newline}(And to set the MINUIT2_LIB and MINUIT2_INC environment variables.) ${newline}This is not generally a problem.${newline}${newline})
    $(warning)
    ALL=
else
    ALL=minuit_wrapper.so
endif


all: $(ALL)

minuit_wrapper.so: minuit_wrapper.cpp
	${CXX} ${CXXFLAGS} -I${MINUIT2_INC} -L${MINUIT2_LIB} -shared -o minuit_wrapper.so minuit_wrapper.cpp -lMinuit2

clean:
	rm -rf minuit_wrapper.so minuit_wrapper.so.dSYM
