#include "stdio.h"

typedef struct des_option {
	char * section;
	char * param;
	char * value;
} des_option;

typedef struct des_optionset{
	// array of des_options
	int n;
	des_option ** option;
} des_optionset;

typedef void* (*des_setup_function)(des_optionset *);
des_setup_function load_des_setup(char * library_name, char * function_name);


#define DEFAULT_OPTION_SECTION "config"

des_optionset * des_optionset_init();
void des_optionset_free(des_optionset * options);
void des_optionset_print(FILE * stream, des_optionset * options);
des_optionset * des_optionset_read(const char * filename);
des_optionset * des_optionset_read_stream(FILE * file);

void des_optionset_add(des_optionset * options, const char * section, const char * param, const char * value);
void des_optionset_set(des_optionset * options, const char * section, const char * param, const char * value);
const char * des_optionset_get(des_optionset * options, const char * section, const char * param);
const char * des_optionset_get_default(des_optionset * options, const char * section, const char * param, const char * default_value);
const char * des_optionset_get_body(des_optionset * options, const char * section, const char * param, int quiet);
int des_optionset_get_int(des_optionset * options, const char * section, const char * param, int * value);
int des_optionset_get_int_default(des_optionset * options, const char * section, const char * param, int * value, int default_value);
int des_optionset_get_double(des_optionset * options, const char * section, const char * param, double * value);
int des_optionset_get_double_default(des_optionset * options, const char * section, const char * param, double * value, double default_value);

