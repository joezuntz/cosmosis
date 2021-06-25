include ${COSMOSIS_SRC_DIR}/config/compilers.mk

define newline


endef



ifeq (${MINUIT2_LIB},)
ifneq (clean,$(MAKECMDGOALS))
    $(warning)
    $(warning ${newline} ${newline}Not building the minuit sampler as the you need the Minuit2 library ${newline}(And to set the MINUIT2_LIB and MINUIT2_INC environment variables.) ${newline}This is not generally a problem.${newline}${newline})
    $(warning)
endif
    ALL=
else
    ALL=minuit_wrapper.so
endif


all: $(ALL)

minuit_wrapper.so: minuit_wrapper.cpp
	-${CXX} ${CXXFLAGS} -I${MINUIT2_INC} -c  minuit_wrapper.cpp
	-${CXX} ${CXXFLAGS} -L${MINUIT2_LIB} -shared -o $(CURDIR)/minuit_wrapper.so minuit_wrapper.o -lMinuit2
	test -f $(CURDIR)/minuit_wrapper.so || echo "\n\nMINUIT sampler did not compile and will not be available\n\n"

clean:
	rm -rf minuit_wrapper.so minuit_wrapper.so.dSYM minuit_wrapper.o
