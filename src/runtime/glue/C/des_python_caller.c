#include "des_python_caller.h"
#include "internal_fits.h"


static void split_path_file(char** p, char** f, const char *pf) {
    char *next;
	const char * slash = pf;
    while ((next = strpbrk(slash + 1, "\\/"))) slash = next;
    if (pf != slash) slash++;
	int sz = slash - pf;
	*p = (char*)malloc(sizeof(char)*(sz+1));
	strncpy(*p, pf, sz);
	(*p)[sz]='\0';
    // *p = strndup(pf, slash - pf);
    *f = strdup(slash);
}


static void prepend_to_python_path(char * dirname)
{
	PyObject *path = PySys_GetObject("path");  //No need to decref this.
	PyObject *dirname_string = PyString_FromString(dirname);
	PyList_Insert(path, 0, dirname_string);
	Py_DECREF(dirname_string);
	
}

static void remove_first_python_path_item()
{
	PyObject *path = PySys_GetObject("path");  //No need to decref this.
	PyList_SetSlice(path, 0, 1, NULL);
}

static char *remove_extension(char* mystr) {
    char *retstr;
    if (mystr == NULL)
         return NULL;
    if ((retstr = malloc (strlen (mystr) + 1)) == NULL)
        return NULL;
    strcpy (retstr, mystr);
    char *lastdot = strrchr (retstr, '.');
    if (lastdot != NULL)
        *lastdot = '\0';
    return retstr;
}

void * des_load_python_interface(char *module_path, char *function_name){

	/* Prepare python. */
	Py_Initialize();

	/* Check that the file path we have been given exists and is readable */
	if (access(module_path, R_OK)){
		fprintf(stderr,"Path %s is non-existent or inaccessible.\n",module_path);
		return NULL;
	}
	
	/* Split the path into the directory and the filename*/
	char * dirname = NULL;
	char * filename = NULL;
	split_path_file(&dirname, &filename, module_path);
	/* Add the directory to the start of python's path so that the module will be found there.*/
	prepend_to_python_path(dirname);
	free(dirname);
	char * module_name = remove_extension(filename);
	free(filename);

	/* Import the function from the named module*/
	PyObject *module = PyImport_ImportModule(module_name);

	/* Reset ourselves to where we were by removing the first path item*/
	remove_first_python_path_item();
	
	/* Check for any errors */
	if (module==NULL) {
		fprintf(stderr,"Failed to import python module '%s'\n",module_name);
		if (PyErr_Occurred()) {
			fprintf(stderr,"Error was:\n"); 
			PyErr_Print();
		}
		else{
			fprintf(stderr,"Error unknown\n"); 
		}
		free(module_name);
		return NULL;
	}
	
	/* Now extract the function from the module */
	PyObject *function = PyObject_GetAttrString(module, function_name);
	Py_DECREF(module);
	if (function==NULL) {
		fprintf(stderr,"Failed to import python function '%s' from module '%s'\n",function_name, module_name);
		free(module_name);
		if (PyErr_Occurred()) {
			fprintf(stderr,"Error was:\n"); 
			PyErr_Print();
		}
		else{
			fprintf(stderr,"Error unknown\n"); 
		}
		return NULL;
	}

	if (!PyCallable_Check(function)){
		fprintf(stderr,"Python object '%s' (from module '%s') is not a callable funcion\n",function_name, module_name);
		free(module_name);
		return NULL;
	}
	
	return (void*)function;
}

int des_call_python(PyObject * function, size_t handle){
	
    /* Call the function with the argument n */
	/* Convert the integer into a python int object*/
    PyObject *arglist = Py_BuildValue("(l)", handle);
	/* Call the function with the object*/
    PyObject *result = PyEval_CallObject(function, arglist);
	/* Free the memory of the int object*/
    Py_DECREF(arglist);

	/* Check result is integer*/
	if (result==NULL || !PyInt_Check(result) ){
		fprintf(stderr,"Unable to parse any integer return value of function called with arg %ld\n", handle);
		if (result) Py_DECREF(result);
		if (PyErr_Occurred()) {
			fprintf(stderr,"Error was:\n"); 
			PyErr_Print();
		}
		else{
			fprintf(stderr,"Error unknown\n"); 
		}
		
		return 4;
	}
	/* Convert the result int object back into an integer.*/
	int result_value = (int) PyInt_AsLong(result);
	/* Free the memory of the result object*/
	Py_DECREF(result);
	
	/* All done!*/
	return result_value;
}

int des_call_python_by_name(char *module_path, char *function_name, size_t n){
		
	PyObject * function = (PyObject*) des_load_python_interface(module_path, function_name);
	if (!function) return 1;
	int result = des_call_python(function, n);
    Py_DECREF(function);
	return result;
}
