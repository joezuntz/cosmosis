include ${COSMOSIS_SRC_DIR}/config/compilers.mk

minuit_wrapper.so: minuit_wrapper.cpp
	${CXX} ${CXXFLAGS} -shared -o minuit_wrapper.so minuit_wrapper.cpp -lMinuit2

clean:
	rm -rf minuit_wrapper.so minuit_wrapper.so.dSYM