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
int cosmosis_minuit2_wrapper(
	int nparam, 
	double * start, 
	likelihood_function f, 
	unsigned int max_evals,
	const char ** param_names
	)
{

	CosmosisMinuitFunction func(f);

	// Set up all the parameters required
	MnUserParameters upar;
	for (int i=0; i<nparam; i++){
		upar.Add(param_names[i], start[i], 0.1);
		// Since we will pass in normalized parameters
		// the limits are all (0,1)
	    upar.SetLimits(param_names[i], 0.0, 1.0);
	}

	MnMigrad migrad(func, upar);
	FunctionMinimum min = migrad(max_evals, 0.01);

	std::cout << "MINUIT convergence information:" << min;

	for (int i=0; i<nparam; i++){
		start[i] = min.UserState().Value(i);
	}
	int status = min.IsValid() ? 0 : 1;
	return status;
}

}