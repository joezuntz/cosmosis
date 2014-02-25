#include "des_options.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "ini.h"

#define MAX_OPTIONS 4096

des_optionset * des_optionset_init()
{
	des_optionset * options = (des_optionset*) malloc(sizeof(des_optionset));
	options->n = 0;
	options->option = (des_option**)malloc(sizeof(des_option*)*MAX_OPTIONS);
	return options;
}

void des_option_free(des_option * option){
	free(option->section);
	free(option->param);
	free(option->value);
	free(option);
}

void des_optionset_free(des_optionset * options)
{
	int i;
	for (i=0; i<options->n; i++){
		des_option_free(options->option[i]);
	}
	options->n = 0;
	free(options->option);
	options->option = NULL;
	free(options);
}




static des_option * des_option_create(const char * section, const char * param, const char * value)
{
	des_option * opt = (des_option*) malloc(sizeof(des_option));
	opt->section = strdup(section);
	opt->param = strdup(param);
	opt->value = strdup(value);
	return opt;
}


des_optionset * des_optionset_read(const char * filename)
{
	FILE * file = fopen(filename, "r");
	if (file==NULL){
		fprintf(stderr, "Ini file %s could not be found or opened.\n", filename);
		return NULL;
	}
	des_optionset * options = des_optionset_read_stream(file);
	if (options==NULL){
		fprintf(stderr, "The problematic ini file was %s\n",filename);
	}
	return options;

}

typedef int (*inih_handler)(void* user, const char* section, const char* name, const char* value);


static int des_optionset_inih_handler(void* user, const char* section, const char* name, const char* value){
	des_optionset * options = (des_optionset*) user;
	des_optionset_add(options, section, name, value);
}


des_optionset * des_optionset_read_stream(FILE * file)
{
	des_optionset * options = des_optionset_init();
	int status = ini_parse_file(file, des_optionset_inih_handler, (void*) options);
	if (status>0){
		fprintf(stderr, "Could not parse ini file: there was a problem on line %d\n",status);
		des_optionset_free(options);
		return NULL;
	}
	else if (status<0){
		fprintf(stderr, "Could not parse ini file: I was not able to read the file for some reason (status=%d)\n",status);
		des_optionset_free(options);
		return NULL;
	}
	return options;

}

   // For each name=value pair parsed, call handler function with given user
   // pointer as well as section, name, and value (data only valid for duration
   // of handler call). Handler should return nonzero on success, zero on error.

   // Returns 0 on success, line number of first error on parse error (doesn't
   // stop on first error), -1 on file open error, or -2 on memory allocation



void des_optionset_add(des_optionset * options, const char * section, const char * param, const char * value)
{
	des_option * opt = des_option_create(section, param, value);
	options->option[options->n] = opt;
	options->n++;
}

void des_optionset_print(FILE * stream, des_optionset * options){
	char * current_section = NULL;
	int i;
	for (i=0; i<options->n; i++){
		des_option * option = options->option[i];

		// New section
		if (current_section==NULL || strcmp(current_section, option->section)){
			if (current_section!=NULL) fprintf(stream, "\n");
			fprintf(stream, "[%s]\n", option->section);
			current_section=option->section;
		}

		fprintf(stream, "%s = %s\n", option->param, option->value);
	}
}

static int des_optionset_find(des_optionset * options, const char * section, const char * param)
{
	int i;
	for (i=0; i<options->n; i++){
		des_option * opt = options->option[i];
		if (  0==strcasecmp(opt->section, section)
			& 0==strcasecmp(opt->param  , param  ) ) return i;
	}
	return options->n;
}

void des_optionset_set(des_optionset * options, const char * section, const char * param, const char * value)
{
	int i = des_optionset_find(options, section, param);
	if (i<options->n) des_option_free(options->option[i]);
	options->option[i] = des_option_create(section, param, value);
}

const char * des_optionset_get_body(des_optionset * options, const char * section, const char * param, int quiet)
{
	int i = des_optionset_find(options, section, param);
	if (i==options->n){
		if (!quiet){
			fprintf(stderr, "Option '%s' not found in section '%s'.\n", param, section);
		}
		return NULL;
	}
	return options->option[i]->value;
}


const char * des_optionset_get(des_optionset * options, const char * section, const char * param)
{
	int quiet=0;
	return des_optionset_get_body(options, section, param, quiet);
}



const char * des_optionset_get_default(des_optionset * options, const char * section, const char * param, const char * default_value)
{
	int quiet=1;
	const char * value = des_optionset_get_body(options, section, param, quiet);
	if (value==0) value=default_value;
	return value;
}


int des_optionset_get_int(des_optionset * options, const char * section, const char * param, int * value)
{
	const char * value_string = des_optionset_get(options, section, param);
	if (value_string==NULL) {
		*value=0;
		return 1;
	}
	char *e = NULL;
	int base=10;
	long int value_int = strtol(value_string, &e, base);

	if (e == value_string) {
		fprintf(stderr,
			"Could not parse parameter '%s' in section '%s' with value '%s' into the expected integer.\n", 
			param, section, value_string);
		fprintf(stderr, "Returning -1; this may cause a crash.\n");
		*value = -1;
		return 2;
	}

	// Otherwise everything is fine.
	*value = value_int;
	return 0;
}



int des_optionset_get_int_default(des_optionset * options, const char * section, const char * param, int * value, int default_value)
{
	int quiet=1;
	const char * value_string = des_optionset_get_body(options, section, param, quiet);
	if (value_string==NULL) {
		*value = default_value;
		return 0;
	}
	char *e = NULL;
	int base=10;
	long int value_int = strtol(value_string, &e, base);

	if (e == value_string) {
		fprintf(stderr,
			"Could not parse parameter '%s' in section '%s' with value '%s' into the expected integer.\n", 
			param, section, value_string);
		fprintf(stderr, "Returning default value %d; this may not be what you want.\n",default_value);
		*value = default_value;
		return 2;
	}

	// Otherwise everything is fine.
	*value = value_int;
	return 0;
}







int des_optionset_get_double(des_optionset * options, const char * section, const char * param, double * value)
{
	const char * value_string = des_optionset_get(options, section, param);
	if (value_string==NULL) {
		*value=0.0;
		return 1;
	}
	char *e = NULL;
	double value_double = strtod(value_string, &e);

	if (e == value_string) {
		fprintf(stderr,
			"Could not parse parameter '%s' in section '%s' with value '%s' into the expected double.\n", 
			param, section, value_string);
		fprintf(stderr, "Returning NaN; this may cause a crash.\n");
		*value = nan(NULL);
		return 2;
	}

	// Otherwise everything is fine.
	*value = value_double;
	return 0;
}



int des_optionset_get_double_default(des_optionset * options, const char * section, const char * param, double * value, double default_value)
{
	int quiet=1;
	const char * value_string = des_optionset_get_body(options, section, param, quiet);
	if (value_string==NULL) {
		*value = default_value;
		return 0;
	}
	char *e = NULL;
	double value_double = strtod(value_string, &e);

	if (e == value_string) {
		fprintf(stderr,
			"Could not parse parameter '%s' in section '%s' with value '%s' into the expected double.\n", 
			param, section, value_string);
		fprintf(stderr, "Returning default value %lf; this may not be what you want.\n",default_value);
		*value = default_value;
		return 2;
	}

	// Otherwise everything is fine.
	*value = value_double;
	return 0;
}



