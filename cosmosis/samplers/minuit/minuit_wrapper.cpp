#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include <vector>
#include <iostream>

using namespace ROOT;
using namespace Minuit2;

typedef double (*likelihood_function)(double * params);

class CosmosisMinuitFunction : public FCNBase {
public:
	CosmosisMinuitFunction(likelihood_function f);
	virtual double Up() const;
	virtual double operator()(const std::vector<double>&) const;
private:
	likelihood_function cFunctionPointer;

};

CosmosisMinuitFunction::CosmosisMinuitFunction(likelihood_function f){
	cFunctionPointer = f;
}



double CosmosisMinuitFunction::Up() const {
	return 1.0;
}

double CosmosisMinuitFunction::operator()(const std::vector<double>& x) const
{
	int n = x.size();
	double p[n];
	for (int i=0; i<n; i++) p[i] = x[i];
	return cFunctionPointer(p);
}

extern "C" {
int cosmosis_minuit2_wrapper(int nparam, double * start, likelihood_function f)
{

	CosmosisMinuitFunction func(f);

	MnUserParameters upar;
	for (int i=0; i<nparam; i++){
		char name[8];
		snprintf(name, 8, "p_%d", i);
		upar.Add(name, start[i], 0.05);
	    upar.SetLimits(name, 0.0, 1.0);
	}

	MnMigrad migrad(func, upar);
	FunctionMinimum min = migrad();

	std::cout << "Minimum =  " << min << std::endl;

	for (int i=0; i<nparam; i++){
		start[i] = min.UserState().Value(i);
	}	
	return 0;
}

}