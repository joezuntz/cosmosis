#ifndef _H_GLUE_C_C_PIPELINE
#define _H_GLUE_C_C_PIPELINE
#include "internal_fits.h"

typedef struct des_pipeline{
	int n;  //Number of functions in the pipeline.
	void ** modules; //An array of either PyObject*'s or des_interface's
	int *types; //An array of values of the DES_FUNCTION_TYPE_* below
} des_pipeline;

des_pipeline * des_pipeline_load(int n, char * paths[], char * function_names[]);
void des_pipeline_destroy(des_pipeline* pipeline);
int des_pipeline_run(des_pipeline * pipeline, internal_fits * handle);
des_pipeline * des_pipeline_read(const char * filename);

#define DES_FUNCTION_TYPE_FAILED 0
#define DES_FUNCTION_TYPE_PYTHON 1
#define DES_FUNCTION_TYPE_SHARED 2

#endif
