include ${COSMOSIS_SRC_DIR}/config/compilers.mk

all: minuit_wrapper.so

minuit_wrapper.so: minuit_wrapper.cpp
	${CXX} ${CXXFLAGS} -I${MINUIT2_INC} -L${MINUIT2_LIB} -shared -o minuit_wrapper.so minuit_wrapper.cpp -lMinuit2

clean:
	rm -rf minuit_wrapper.so minuit_wrapper.so.dSYM
