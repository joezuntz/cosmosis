#include <julia.h>
#include <stdio.h>
#include "../../datablock/c_datablock.h"

int cosmosis_julia_is_initialized = 0;



typedef struct julia_module_info {
    char * execute;
    char * setup;
    char * cleanup;
    char * name;
    int debug_mode;
} julia_module_info;



static 
void backtrace(const char * location, const char * name){
    jl_value_t * exception = jl_exception_occurred();
    fprintf(stderr, "\nERROR:\nException ocurred during %s in %s:\n"
        "%s: \n", location, name, jl_typeof_str(exception));

    jl_function_t *println = jl_get_function(jl_base_module, "println");
    jl_call1(println, exception);

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

    // Initialize
    jl_init();

    // Add the path to the cosmosis modules
    char cmd[256];
    snprintf(cmd, 256, "push!(LOAD_PATH, \"%s/datablock/julia\")", cosmosis_src_dir);

    fflush(stderr); 
    jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("path manipulation 1 (please report)", "init");
        return 1;
    }

    // Import the cosmosis julia module
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
load_module(const char * directory, const char * module_name, int debug_mode)
{
    int status = cosmosis_init_julia();

    if (status) return NULL;

    char cmd[512];
    
    snprintf(cmd, 512, "include(\"%s/%s.jl\")", directory, module_name);
    jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("import of module (please report)", module_name);
        return NULL;
    }

    snprintf(cmd, 512, "%s_setup, %s_execute, %s_cleanup = cosmosis.make_julia_module(%s)", module_name, module_name, module_name, module_name);

    jl_eval_string(cmd);

    if (jl_exception_occurred()) {
        backtrace("construction and wrapping of julia module", module_name);
        return NULL;
    }

    // if (debug_mode){
    //     printf("Using debug mode in Julia modules - execute functions will print stack trace on error\n");
    //     snprintf(cmd, 512, "%s_execute = cosmosis.stack_tracer_wrapper(%s.execute)", module_name, module_name);
    // }
    // else{
    //     snprintf(cmd, 512, "%s_execute = %s.execute", module_name, module_name);
    // }
    
    // jl_value_t * execute = jl_eval_string(cmd);

    // if (jl_exception_occurred()) {
    //     backtrace("execute function import", module_name);
    //     return NULL;
    // }


    // snprintf(cmd, 512, "%s_cleanup = cosmosis.stack_tracer_wrapper(%s.cleanup)",module_name,  module_name);
    // jl_eval_string(cmd);
    // int have_cleanup = (jl_exception_occurred()==NULL);
    // jl_exception_clear();


    julia_module_info * info = malloc(sizeof(julia_module_info));
    info->name = strdup(module_name);
    snprintf(cmd, 512, "%s_execute", module_name);
    info->execute = strdup(cmd);
    snprintf(cmd, 512, "%s_setup", module_name);
    info->setup = strdup(cmd);
    snprintf(cmd, 512, "%s_cleanup", module_name);
    info->cleanup = strdup(cmd);
    info->debug_mode = debug_mode;


    return info;
}


void * run_setup(julia_module_info * info, void * options)
{
    jl_value_t * options_jl = jl_box_voidpointer(options);
    jl_value_t * setup = jl_eval_string(info->setup);
    jl_value_t * config = jl_call1(setup, options_jl);
    jl_value_t * ex = jl_exception_occurred();

    if (ex){
        fprintf(stderr, "\nError in setup function:\n~~~~~~~~~~\n");
        jl_static_show(jl_stderr_stream(), ex);
        fprintf(stderr, "\n~~~~~~~~~~\n");
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
    jl_value_t * execute = jl_eval_string(info->execute);
    jl_value_t * status_jl = jl_call(execute, args, 2);


    if (jl_exception_occurred()){
        // If we are in debug mode then a stack trace has already been printed.
        // Otherwise print an error message suggesting they enable it.
        if (!info->debug_mode){
            fprintf(stderr, "Error in Julia execution. Set verbosity to 'noisy' or 'debug' for more stack trace\n");
        }
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
    jl_value_t * cleanup = jl_eval_string(info->cleanup);
    jl_call1(cleanup, config_jl);

    if (jl_exception_occurred()){
        return 1;
    }

    return 0;
}
