#ifndef _H_C_PYTHON_CALLER
#define _H_C_PYTHON_CALLER
#include "Python.h"
#include "stdio.h"
int des_call_python_by_name(char *module_path, char *function_name, size_t n);
void * des_load_python_interface(char *module_path, char *function_name);
int des_call_python(PyObject * function, size_t handle);

#endif