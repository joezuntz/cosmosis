#include <julia.h>
#include <stdio.h>
#include "cosmosis/datablock/c_datablock.h"

int cosmosis_julia_is_initialized = 0;

typedef struct julia_module_info {
    void * setup;
    void * execute;
    void * cleanup;
    char * name;
} julia_module_info;


static 
void backtrace(const char * location, const char * name){
    jl_value_t * exception = jl_exception_occurred();
    fprintf(stderr, "\nERROR:\nException ocurred during %s in %s:\n"
        "%s: \n", location, name, jl_typeof_str(exception));

    jl_function_t *println = jl_get_function(jl_base_module, "println");
    jl_call1(println, exception);

    fprintf(stderr, "Backtrace (actual error is usually buried between some sets jl_ functions):\n");
    fprintf(stderr, "************************\n");
    jlbacktrace();
    fprintf(stderr, "************************\n\n");
    jl_exception_clear();

}


static 
int cosmosis_init_julia(){
    if (cosmosis_julia_is_initialized) return 0;

    // Use the env var to get the path
    char * cosmosis_src_dir = getenv("COSMOSIS_SRC_DIR");

    if (cosmosis_src_dir==NULL){
        fprintf(stderr, "Could not find COSMOSIS_SRC_DIR env var\n");
        return 1;
    }

    printf("Initializing Julia\n");

    // Initialize
    jl_init();

    // Import the cosmosis Julia
    char cmd[256];
    
    snprintf(cmd, 256, "push!(LOAD_PATH, \"%s/cosmosis/datablock/julia\")", cosmosis_src_dir);
    jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("path manipulation 1 (please report)", "init");
        return 1;
    }


    snprintf(cmd, 256, "import cosmosis");
    jl_eval_string("import cosmosis");

    if (jl_exception_occurred()) {
        backtrace("import of cosmosis.jl (please report)", "init");
        return 2;
    }


    // Mark success with global
    cosmosis_julia_is_initialized = 1;

    return 0;
}


julia_module_info * 
load_module(const char * directory, const char * module_name)
{
    int status = cosmosis_init_julia();

    if (status) return NULL;

    char cmd[256];
    
    snprintf(cmd, 256, "push!(LOAD_PATH, \"%s\")", directory);
    jl_eval_string(cmd);
    if (jl_exception_occurred()) {
        backtrace("path manipulation 2 (please report)", module_name);
        return NULL;
    }

    snprintf(cmd, 256, "import %s", module_name);
    jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("import of module (please report)", module_name);
        return NULL;
    }

    snprintf(cmd, 256, "%s.setup", module_name);
    jl_value_t * setup = jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("setup function import", module_name);
        return NULL;
    }

    snprintf(cmd, 256, "%s.execute", module_name);
    jl_value_t * execute = jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("execute function import", module_name);
        return NULL;
    }

    snprintf(cmd, 256, "%s.cleanup", module_name);
    jl_value_t * cleanup = jl_eval_string(cmd);
    jl_exception_clear();


    julia_module_info * info = malloc(sizeof(julia_module_info));
    info->name = strdup(module_name);
    info->setup = setup;
    info->execute = execute;
    info->cleanup = cleanup;

    return info;


}
    
void * run_setup(julia_module_info * info, void * options)
{

    jl_value_t * options_jl = jl_box_voidpointer(options);
    jl_value_t * config = jl_call1(info->setup, options_jl);

    if (jl_exception_occurred()){
        backtrace("setup function call", info->name);
        return NULL;
    }

    if (config==NULL){
        fprintf(stderr, "Note: setup function in %s did not return a value - this may be a mistake.\n", info->name);
        return NULL;
    }

    return (void*)config;
}


int run_execute(julia_module_info * info, void * block, void * config)
{

    jl_value_t * block_jl = jl_box_voidpointer(block);
    jl_value_t * config_jl = (jl_value_t*) config;

    jl_value_t * args[2] = {block_jl, config_jl};
    jl_value_t * status_jl = jl_call(info->execute, args, 2);
    

    if (jl_exception_occurred()){
        backtrace("execute function call", info->name);
        return 1;
    }

    if (status_jl==NULL){
        fprintf(stderr, "Execute function in %s did not return a value - must return an integer status.\n", info->name);
        return 1;
    }

    int status = 2;

    if (jl_typeis(status_jl, jl_int32_type)){
        status = (int) jl_unbox_int32(status_jl);
    }
    else if (jl_typeis(status_jl, jl_int64_type)){
        status = (int) jl_unbox_int64(status_jl);

    }
    else {
        fprintf(stderr, "Execute funtion returned a non-integer.  Must return an int32 status\n");
        return 2;
    }

    return status;
}

int run_cleanup(julia_module_info * info, void * config)
{

    if (info->cleanup==NULL) return 0;
    jl_value_t * config_jl = (jl_value_t*) config;

    jl_call1(info->cleanup, config_jl);

    if (jl_exception_occurred()){
        backtrace("cleanup function call", info->name);
        return 1;
    }

    return 0;
}
