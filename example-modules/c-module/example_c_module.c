#include "cosmosis/datablock/c_datablock.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/* 

	Trivial single parameter likelihood.

	On setup, reads from section "example" 
	values 

*/

typedef struct example_data {
	double omega;
	double sigma;
} example_data;

void * setup(c_datablock * options)
{
	//Ideally need a way of finding the section corresponding to this 
	//particular option set?  Maybe get passed in a section name??
	// Or could save something to the block.
	//At the moment just assume it is called "example".
	const char * section = "example";

	// Allocate space for returned config data
	DATABLOCK_STATUS status=0;
	example_data * data = (example_data *)malloc(sizeof(example_data));

	if ( data == NULL ) {
		fprintf(stderr, "Error allocating memory in example setup.\n");
		exit(1);
	}

	// Read required parameters from options
	status |= c_datablock_get_double(options, section, "measured_omega", &(data->omega));
	status |= c_datablock_get_double(options, section, "measured_error", &(data->sigma));

	// Check for errors and die if failure
	if (status){
		fprintf(stderr,"Need to specify measured_omega and measured_error in options (%d)\n",status);
		exit(status);
	}

	//Return our data - execute will get it back later.
	return (void*) data;
}


int execute(c_datablock * block, void * config)
{
	//Get back our config data
	example_data * data = (example_data*) config;

	//We will be reading one parameter, omega, and doing
	//a likelihood on it
	DATABLOCK_STATUS status=0;
	double omega = 0.0;

	// Load the parameter we want from the datablock
	status = c_datablock_get_double(block, "COSMOLOGY","omega", &omega);

	// Could just do one check for any error in this simple case
	// We should provide a nice macro for this
	if (status) {
		fprintf(stderr, "Required parameter omega not found in COSMOLOGY.  Status: %d\n",status);
		return status;
	}

	// Calculate a likelihood
	double like = -0.5 * pow((omega - data->omega)/data->sigma, 2);

	// Save the likelihood
	status = c_datablock_put_double(block, "LIKELIHOODS", "omega_like", like);

	// More error checking
	if (status) {
		fprintf(stderr, "Failed to save omega_like to datablock in LIKELIHOOD.  Status: %d\n",status);
		return status;
	}
	//Signal success.
	return 0;
}

void cleanup(void * config)
{
	// Simple tidy up - just free what we allocated in setup
	example_data * data = (example_data*) config;
	free(data);
}
