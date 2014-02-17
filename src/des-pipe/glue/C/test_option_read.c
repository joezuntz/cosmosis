#include "des_options.h"


int main(){
	des_optionset * options = des_optionset_read("my.ini");
	if (options==NULL) return 1;


	const char * model = des_optionset_get(options, "camb", "model");
	printf("Got model = %s\n", model);


	double mean = 0.0;
	int status = des_optionset_get_double(options, "H0", "mean", &mean);
	printf("Got mean = %lf\n", mean);

	int N;
	status |= des_optionset_get_int_default(options, "H0", "N", &N, 42);
	printf("Got N = %d\n", N);


	return status;
}