#include "clik.h"
#include "internal_fits.h"
#include "des_options.h"


typedef struct configuration_data{
	int ready;
	clik_object * T_low_data;
	clik_object * P_low_data;
	clik_object * T_high_data;
	clik_object * lensing_data;


} configuration_data;


static int find_max_lmax(clik_object * like_obj)
{
	error * err = initError();
	int n_cl = 6;
	int lmax_by_type[n_cl];
	clik_get_lmax(like_obj, lmax_by_type,&err);
	int max_lmax = 0;
	int i;
	for (i=0; i<n_cl; i++) {
		if (lmax_by_type[i]>max_lmax){
			max_lmax = lmax_by_type[i];
		}
	}
	return max_lmax;
	endError(&err);
}



configuration_data * setup(des_optionset * options){
	error * err = initError();

	const char * T_low_file = des_optionset_get_default(options, DEFAULT_OPTION_SECTION, "t_low_file", (const char*)NULL);
	const char * P_low_file = des_optionset_get_default(options, DEFAULT_OPTION_SECTION, "p_low_file", (const char*)NULL);
	const char * T_high_file = des_optionset_get_default(options, DEFAULT_OPTION_SECTION, "t_high_file", (const char*)NULL);
	const char * lensing_file = des_optionset_get_default(options, DEFAULT_OPTION_SECTION, "lensing_file",(const char*)NULL);

	configuration_data * config = malloc(sizeof(configuration_data));
	config->T_low_data = NULL;
	config->P_low_data = NULL;
	config->T_high_data = NULL;
	config->lensing_data = NULL;

	if (T_low_file) {
		config->T_low_data = clik_init((char*)T_low_file,&err);
	}
	if (P_low_file) {
		config->P_low_data = clik_init((char*)P_low_file,&err);
	}
	if (T_high_file) {
		config->T_high_data = clik_init((char*)T_high_file,&err);
	}

	if (lensing_file) {
		config->lensing_data = clik_init((char*)lensing_file,&err);
	}

	if (T_low_file==NULL && P_low_file==NULL && T_high_file==NULL && lensing_file==NULL){
		fprintf(stderr, "No data files were specified for Planck at all!\n");
		fprintf(stderr, "In the options file you need to set at least one of:\n");
		fprintf(stderr, " - t_low_file\n");
		fprintf(stderr, " - t_high_file\n");
		fprintf(stderr, " - p_low_file\n");
		fprintf(stderr, " - lensing_file\n");
		fprintf(stderr, " To the path to the planck files you can download from:\n");
		fprintf(stderr, " http://pla.esac.esa.int/pla/aio/planckProducts.html\n");
		exit(1);
	}

	if (isError(err)){
		fprintf(stderr,"There was an error initializating the Planck likelihoods.  See below.\n");
		quitOnError(err,__LINE__,stderr);
	}

	endError(&err);

	return config;
}


static 
int run_clik_cosmosis(fitsfile * package, clik_object * like_obj, double * output_like)
{
	int i;
#define N_CL 6
	int param_count=0;
	int status = 0;
	char * package_names[N_CL] = {"TT","EE","BB","TE","TB","EB"};
	int lmax_by_type[N_CL];

	error * err = initError();

	// Get the lmax values for the different spectra and sum them
	clik_get_lmax(like_obj, lmax_by_type,&err);
	for (i=0;i<N_CL;i++) if (lmax_by_type[i]>0) param_count += lmax_by_type[i]+1; 

	if (isError(err)){
		fprintf(stderr,"There was an error getting lmax from a Planck likelihood.\n");
		return 1;
	}


	// Get the number and names of extra parameters and add to total
	parname * nuisance_names;
	int n_nuisance = clik_get_extra_parameter_names(like_obj, &nuisance_names, &err);
	param_count += n_nuisance;

	if (isError(err)){
		fprintf(stderr,"There was an error getting nuisance params from a Planck likelihood.\n");
		return 2;
	}


	// p is the overall space and cl is the space within it for each spectra.
	double * p = malloc(sizeof(double)*param_count);
	double * cl = p;

	// 
	status |= fits_goto_extension(package, CMB_CL_SECTION);
	for (i=0;i<N_CL;i++) {
		// Check lmax for this spectrum.  If not needed, continue
		int lmax = lmax_by_type[i];
		if (lmax<=0) continue;
		// Otherwise fill in first the trivial cl and then the rest, loaded fromt the package
		cl[0]=0.0;
		cl[1]=0.0;
		int nread=0;
		status |= fits_get_double_column_preallocated(package, package_names[i], cl+2, lmax-1, &nread);

		// Check it all worked
		FAIL_AND_CLOSE_ON_STATUS(package, status,"Could not load TT or ell for Planck likelihood");
		if (nread<lmax_by_type[i]-1) status=nread+1;
		FAIL_AND_CLOSE_ON_STATUS(package, status,"Could not get long enough column for TT Planck");

		// Transform to raw cl from l**2 cl
		int ell;
		for (ell=2; ell<=lmax; ell++) cl[ell]/= (ell*(ell+1.0)/2.0/M_PI);

		// Move the spectrum pointer forward ready for the next bit
		cl+=lmax+1;
	}

	// Now move to the planck section to get the nuisance parameters
	status |= fits_goto_extension(package, PLANCK_SECTION);

	for (i=0; i<n_nuisance; i++){
		status|=fits_get_double_parameter(package, nuisance_names[i], 	cl+i  ); //ok
	}
	free(nuisance_names);
	FAIL_AND_CLOSE_ON_STATUS(package, status,"Could not get one or more of the Planck nuisance parameters");


	// Compute the actual likelihood and check for errors
	double like = clik_compute(like_obj, p, &err);
	free(p);

	if (isError(err)){
		status = 1;
		FAIL_AND_CLOSE_ON_STATUS(package, status,"Planck likelihood error");
	}
	endError(&err);	
	*output_like = like;

	return 0;
}
#undef n_cl


int execute(internal_fits * handle, configuration_data * config){

	fitsfile * package = fitsfile_from_internal(handle);

	double tt_high_like = 0.0;
	double tt_low_like = 0.0;
	double p_low_like = 0.0;
	int status = 0;
	if (config->T_high_data) status = run_clik_cosmosis(package, config->T_high_data, &tt_high_like);
	if (config->T_low_data) status |= run_clik_cosmosis(package, config->T_low_data,  &tt_low_like );
	if (config->P_low_data) status |= run_clik_cosmosis(package, config->P_low_data,  &p_low_like  );

	if (status) return status;

	double like = tt_high_like + tt_low_like + p_low_like;
	status |= fits_goto_or_create_extension(package, LIKELIHOODS_SECTION);
	status |= fits_put_double_parameter(package,  "PLANCK_LIKE", like, "Planck likelihood(s)");
	FAIL_AND_CLOSE_ON_STATUS(package, status,"Could not set the Planck like okay");

	status |= close_fits_object(package);









	// Get T data vector from handle


	// Pull planck nuisance parameters out of package Exact vector will depend on likes used

	// Run each of the likelihoods in turn.  

	//Save PLANCK_LIKE

	return 0;
}