#include "des_python_caller.h"
#include "des_pipeline.h"
#include "internal_fits.h"
#include "string.h"
#include "ini.h"


static int string_ends_with(const char * string, const char *suffix){
	int n = strlen(string);
	int s = strlen(suffix);
	if (n<s) return 0;
	for (int i=0; i<s; i++){
		if (string[n-1-i]!=suffix[s-1-i]) return 0;
	}
	return 1;
}

void des_pipeline_destroy(des_pipeline* pipeline){
	free(pipeline->types);
	free(pipeline->modules);
	pipeline->n=0;
	free(pipeline);
	return;
}


des_pipeline * des_pipeline_load(int n, char * paths[], char * function_names[])
{
	// Allocate the pipeline
	des_pipeline * pipeline = (des_pipeline* )malloc(sizeof(des_pipeline));
	pipeline->modules = (void**) malloc(sizeof(void*)*n);
	pipeline->types = (int*) malloc(sizeof(int)*n);
	pipeline->n = n;
	
	//Loop through the paths and function names, loading each in turn.
	for(int i=0; i<n; i++){
		//Get the path and name
		char * path = paths[i];
		char * function_name = function_names[i];

		//Check what kind of filename is used and load using the corresponding function.
		if (string_ends_with(path,".py")){ //Python file
			pipeline->modules[i] = (void*) des_load_python_interface(path, function_name);
			pipeline->types[i] = DES_FUNCTION_TYPE_PYTHON;
		}
		else if (string_ends_with(path,".so") || string_ends_with(path,".dylib")){ //Shared library
			pipeline->modules[i] = (void*) load_des_interface(path, function_name);
			pipeline->types[i] = DES_FUNCTION_TYPE_SHARED;
		}
		else{ //Anything else then we have no idea what to do.
			fprintf(stderr, "load_des_pipeline does not know how to load the file called %s.  Was expecting a filename ending in .py, .so, or .dylib", path);
			pipeline->modules[i] = NULL;
			pipeline->types[i] = DES_FUNCTION_TYPE_FAILED;
		}
	}
	
	// Check for any failed loads and if so return NULL;
	for(int i=0; i<n; i++){
		if (pipeline->modules[i]==NULL){
			fprintf(stderr, "Failed to load one or more modules.  Returning NULL.\n");
			des_pipeline_destroy(pipeline);
			pipeline = NULL;
			return pipeline;
		}
	}
	
	return pipeline;
}

int des_pipeline_run_element(void * element, int type, internal_fits * handle){
	//Run a single module in the pipeline - library or python
	int status;
	if (type==DES_FUNCTION_TYPE_PYTHON){  	// Run a python module function
		status = des_call_python(element,(size_t)handle);
	} 
	else if (type==DES_FUNCTION_TYPE_SHARED){ // Run a shared library function
		des_interface_function function = (des_interface_function) element;
		status = function(handle);
	}
	else{
		fprintf(stderr, "Unknown function type passed to des_pipeline_run_element: %d\n",type);
		status = -1;
	}
	return status;
}

int des_pipeline_run(des_pipeline * pipeline, internal_fits * handle){
	int status = 0;
	for (int i=0; i<pipeline->n; i++){
		status = des_pipeline_run_element(pipeline->modules[i], pipeline->types[i], handle);
		if (status){
			fprintf(stderr,"Got an error status %d in the pipeline at stage %d.\nAborting.\n", status, i);
			return status;
		}
	}
	return status;
	
}

#define MAX_SECTIONS 128
#define MAX_PARAMS 128
#define INI_PIPELINE_SECTION "pipeline"
#define INI_MODULE_ROOT "root"
#define INI_MODULES "modules"
#define INI_FILENAME "file"
#define INI_FUNCTION "function"
#define streq(a,b) (strcmp(a,b)==0)


typedef struct pipeline_section_info{
	char * file;      //Only if this is a module
	char * function;  //Only if this is a module
	int nparam;
	char * params[MAX_PARAMS];
	char * values[MAX_PARAMS];
} pipeline_section_info;


typedef struct pipeline_setup_info{
	char * root;  //Root of module directories
	int current_index;
	char * current_section;
	pipeline_section_info * main_info; //Top-level names, values
	int nmodule; //Number of modules to run.
	char * modules[MAX_SECTIONS];  //Names of modules to run
	int nsection; //Number of sections in ini file.
	char * sections[MAX_SECTIONS]; //Names of sections in in file.
	pipeline_section_info * section_info[MAX_SECTIONS]; 
} pipeline_setup_info;
	
	
static pipeline_setup_info * des_pipeline_setup_info_create(){
	pipeline_setup_info * info = (pipeline_setup_info*) malloc(sizeof(pipeline_setup_info));
	info->root = NULL;
	info->nmodule = 0;
	info->nsection = 0;
	info->current_index = -1;
	info->current_section = NULL;
	for(int i=0;i<MAX_SECTIONS;i++) info->modules[i] = NULL;
	for(int i=0;i<MAX_SECTIONS;i++){
		info->section_info[i] = (pipeline_section_info*)malloc(sizeof(pipeline_section_info));
		info->section_info[i]->file = NULL;
		info->section_info[i]->function = NULL;
		info->section_info[i]->nparam = 0;
		for(int j=0;j<MAX_PARAMS;j++){
			info->section_info[i]->params[j] = NULL;
			info->section_info[i]->values[j] = NULL;
		}
	}
	info->main_info = (pipeline_section_info*)malloc(sizeof(pipeline_section_info));
	info->main_info->file = NULL;
	info->main_info->function = NULL;
	info->main_info->nparam = 0;
	for(int j=0;j<MAX_PARAMS;j++){
		info->main_info->params[j] = NULL;
		info->main_info->values[j] = NULL;
	}
	return info;	
}

static void des_destroy_section_info(pipeline_section_info * section_info){
	free(section_info->file);
	free(section_info->function);
	for(int j=0;j<MAX_PARAMS;j++){
		free(section_info->params[j]);
		free(section_info->values[j]);
	}
}

static void des_pipeline_setup_info_destroy(pipeline_setup_info * info){
	for(int i=0;i<MAX_SECTIONS;i++){
		des_destroy_section_info(info->section_info[i]);
	}	
	des_destroy_section_info(info->main_info);
	for(int i=0;i<info->nmodule;i++){
		free(info->modules[i]);
	}
	for(int i=0;i<info->nsection;i++){
		free(info->sections[i]);
	}
		
	free(info);
}

static int des_pipeline_ini_parse_pipeline_section(pipeline_setup_info * info, const char * name, const char * value){
	//Module root directory
	if (streq(name,INI_MODULE_ROOT)){
		info->root = strdup(value);
	}
	//List of modules.
	//List of modules to run, split by whitespace.
	//Parse the number of them into info->nmodule and the list into info->modules.
	//Otherwise the section must refer to a particular module.
	//Any module listed must be a section listed later.
	else if (streq(name,INI_MODULES)){
		char * module_list = strdup(value);
		char * module;
		info->nmodule=0;
		for (int i=0; i<MAX_SECTIONS;i++) info->modules[i]=NULL;
		while(module = strsep(&module_list, ", \t")){
			if (strlen(module)==0) continue;
			info->modules[info->nmodule] = strdup(module);
			info->nmodule++;
		}
		free(module_list);
	}
	// Otherwise this is a higher level parameter for the pipeline or sampler.
	// Just save the name and value.
	else{
		info->main_info->params[info->main_info->nparam] = strdup(name);
		info->main_info->values[info->main_info->nparam] = strdup(value);
		info->main_info->nparam++;
	}
	return 0;
	
}

int des_pipeline_ini_parse_module_section(pipeline_setup_info * info, const char * section, const char * name, const char * value){

	//New section
	if (info->current_section==NULL || strcmp(section,info->current_section)){
		info->current_index++;
		//Record the new section name, and the index and name it has.
		info->sections[info->current_index] = strdup(section);
		info->current_section = info->sections[info->current_index];
		info->nsection++;
	}
	
	pipeline_section_info * section_info = info->section_info[info->current_index];
	//Two special parameter names are "file" and "function", which set the obvious things
	if (streq(name,INI_FILENAME)){
		section_info->file = strdup(value);
	}
	else if (streq(name,INI_FUNCTION)){
		section_info->function = strdup(value);
	}
	else{
		int np = section_info->nparam;
		section_info->params[np] = strdup(name);
		section_info->values[np] = strdup(value);
		section_info->nparam++;
	}
	return 0;
	
}

static int des_pipeline_ini_handler(void * user, const char * section, const char * name, const char * value){
	pipeline_setup_info * info = (pipeline_setup_info*)user;
	int status = 0;
	//If section is pipeline - record top-level information
	if (streq(section,INI_PIPELINE_SECTION)){
		status = des_pipeline_ini_parse_pipeline_section(info, name, value);
	}
	//Otherwise it is a module section.
	else {
		status = des_pipeline_ini_parse_module_section(info, section, name, value);
	}
	//The inih library wants nonzero for success.
	status = !status;
}

static char * root_path(const char * c1, const char * c2){
	int n1 = strlen(c1);
	int n = n1 + strlen(c2);
	char * c3 = malloc((n+2)*sizeof(char));
	strcpy(c3, c1);
	c3[n1]='/';
	c3[n1+1]='\0';
	strcat(c3, c2);
	return c3;
}

des_pipeline * des_pipeline_from_info(pipeline_setup_info * info){
	int n_pipe = info->nmodule;
	char * paths[n_pipe];
	char * function_names[n_pipe];
	for (int p=0; p<n_pipe;p++){
		char * pipe_name = info->modules[p];
		pipeline_section_info * section_info = NULL;
		//Find the index of the module in the sections
		//and a pointer to the corresponding info
		
		for (int i=0; i<info->nsection; i++){
			if (streq(pipe_name,info->sections[i])){
				section_info = info->section_info[i];
				break;
			}	
		}
		//Could not find the section
		if (section_info==NULL){
			fprintf(stderr,"Could not find a section in the ini file about the module called %s.  Unable to load.\n",pipe_name);
			return NULL;
		}
		paths[p] = root_path(info->root,section_info->file);
		if (paths[p]==NULL){
			fprintf(stderr,"Could not find a parameter 'file' in the ini file about the module called %s.  Need to specify this to load it.\n",pipe_name);
			return NULL;
		}
		function_names[p] = section_info->function;
		if (function_names[p]==NULL){
			fprintf(stderr,"Could not find a parameter 'function' in the ini file about the module called %s.  Need to specify this to load it.\n",pipe_name);
			return NULL;
		}
	}
	
	des_pipeline * pipeline = des_pipeline_load(n_pipe, paths, function_names);
	for (int p=0; p<n_pipe; p++){
		free(paths[p]);
	}
	return pipeline;
}

des_pipeline * des_pipeline_read(const char * filename){
	pipeline_setup_info * info = des_pipeline_setup_info_create();
	int status = ini_parse(filename, des_pipeline_ini_handler, info);
	if (status) {
		fprintf(stderr,"Error loading pipeline from ini file %s.\nThere was an error on line %d of the file.\n",filename,status);
		des_pipeline_setup_info_destroy	(info);
		return NULL;
	}
	des_pipeline * pipeline = des_pipeline_from_info(info);
	des_pipeline_setup_info_destroy(info);
	return pipeline;
	
	
	
	
}