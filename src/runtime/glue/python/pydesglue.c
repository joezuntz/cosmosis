#include <Python.h>
#include <cStringIO.h>
#include "pydesglue.h"



//Internal function - not called from python
internal_fits * 
convert_string_to_fits(PyObject * string)
{
	
	// Check it is really a string
	if (!PyString_Check(string)) {
			PyErr_SetString(PyExc_TypeError, "Cannot parse non-string object to FITS");
	        return NULL;
	}
	
	//Turn the python string into ordinary objects.
	size_t len = PyString_Size(string);
	if (len==0){
		PyErr_SetString(PyExc_TypeError, "Cannot parse empty string to FITS");
        return NULL;
	}
	//This refers to the internal buffer of the string and
	//must not be modified or freed
	char* buffer = PyString_AsString(string); 

	// Copy the memory as we do not want to modify the python string in-place.
	// That would be very naughty.
	internal_fits * output = malloc(sizeof(output));
	output->file_ptr=malloc(len*sizeof(char));
	memcpy(output->file_ptr,buffer,len*sizeof(char));
	output->file_size=len;
	
	int status=0;
	fitsfile * f;
	fits_open_memfile(&f, "", READWRITE, (void**)&(output->file_ptr), &(output->file_size), 
	                  INTERNAL_FITS_INITIAL_BLOCK_SIZE, realloc, &status);
	if (status) {
		PyErr_SetString(PyExc_ValueError, "Could not parse string to FITS");
		status=0;
		fits_close_file(f,&status);
		free(output);
		return NULL;
	}
	fits_close_file(f,&status);
	return output;
}



//Internal function - not called from python
PyObject * 
convert_fits_to_stringio(internal_fits * F)
{
	PycString_IMPORT;
    PyObject *pybuf = PyBuffer_FromMemory(F->file_ptr, F->file_size);
	PyObject *sio = PycStringIO->NewInput(pybuf);
	return sio;
}







// Python interface stuff

static PyObject *
desglue_create_fits(PyObject *self, PyObject *args)
{
	PyObject * string;
    if (!PyArg_ParseTuple(args, "O", &string))
        return NULL;
	internal_fits * F = convert_string_to_fits(string);
	return PyLong_FromLong((size_t) F);
}

static PyObject * 
desglue_rewrite_fits(PyObject *self, PyObject *args)
{
	/* Rewrite a fits object with a python string.  Does not currently error check to ensure string is really valid FITS.*/
	
	/* Parse the arguments: a fits pointer and a string*/
	PyObject * string;
	long int p;
	if (!PyArg_ParseTuple(args, "lO", &p, &string))
        return NULL;

	/* Get the string length and character buffer*/
	Py_ssize_t string_length;
	char * string_ptr;
	int err = PyString_AsStringAndSize(string, &string_ptr, &string_length);
	if (err){
		PyErr_SetString(PyExc_ValueError, "Non-string passed to desglue rewrite_fits");
		return NULL;    
	}
	
	/* Test for emptry strings */
	if (string_length<=0){
		PyErr_SetString(PyExc_ValueError, "Zero length string passed to desglue_rewrite_fits");
		return NULL;
	}
	
	internal_fits * F = (internal_fits*) p;
	
	/* Test for zero buffer.  Not a very good test - mininal.*/
	if (F->file_ptr==NULL || F->file_size<=0){
		PyErr_SetString(PyExc_ValueError, "Non-fits handle or empty fits passed to desglue_rewrite_fits");
		return NULL;
	}
	
	/* Free up the old data*/
	free(F->file_ptr);
	F->file_ptr = NULL;
	F->file_size = 0;
	
	/* Assign the new data */
	F->file_size = string_length;
	F->file_ptr = malloc(string_length * sizeof(char));
	memcpy(F->file_ptr, string_ptr, string_length*sizeof(char) );
	
	/* Return None */
	Py_INCREF(Py_None);
	PyObject *result=Py_None;
	return result;
	
	
}

static PyObject *
desglue_read_fits(PyObject *self, PyObject *args)
{
	/* This should probably be a size_t.  Need to investigate and understand 32 vs 64 bit issues.  Do we need to support 32 bit? */
	size_t p;
    if (!PyArg_ParseTuple(args, "l", &p))
        return NULL;
	if (p==0) {
		PyErr_SetString(PyExc_ValueError,"Passed a NULL pointer to desglue_read_fits");
		return NULL;
	}
	PyObject * stringio = convert_fits_to_stringio((internal_fits *) p);
	return stringio;
}


static PyObject *
desglue_free_fits(PyObject *self, PyObject *args)
{
	long int p;

    if (!PyArg_ParseTuple(args, "l", &p))
        return NULL;

	delete_fits_object((internal_fits*)p);

    Py_INCREF(Py_None);
	PyObject *result=Py_None;
	return result;
}

static PyObject *
desglue_get_option_from_set(PyObject *self, PyObject *args)
{
	long int optionset;
	char * section_ptr;
	char * param_ptr;

    if (!PyArg_ParseTuple(args, "lss", &optionset, &section_ptr, &param_ptr))
        return NULL;

    int quiet = 1;
	const char * value = des_optionset_get_body((des_optionset *) optionset, 
											section_ptr, param_ptr, quiet);

	if (value==NULL){
		char message[256];
		snprintf(message, 256, "Could not find option %s in section %s", param_ptr, section_ptr);
		PyErr_SetString(PyExc_KeyError, message);
		return NULL;
	}

	return Py_BuildValue("s", value);


}

static PyObject *
desglue_create_optionset_from_iterator(PyObject *self, PyObject *args)
{
	PyObject * parameter_list;
    if (!PyArg_ParseTuple(args, "O", &parameter_list))
        return NULL;


	PyObject *iterator = PyObject_GetIter(parameter_list);
	PyObject *parameter_group;

	if (iterator == NULL) {
		PyErr_SetString(PyExc_ValueError, 
			"When creating an OptionSet you must use a list-like object of (section, param, value) triplets");
		return NULL;

	}

	des_optionset * optionset = des_optionset_init();

	while ((parameter_group = PyIter_Next(iterator))) {
	    /* Get the three parts of the parameter group and add them to the OptionSet */
		PyObject* section = PySequence_GetItem(parameter_group, (Py_ssize_t) 0);
		PyObject* param   = PySequence_GetItem(parameter_group, (Py_ssize_t) 1);
		PyObject* value   = PySequence_GetItem(parameter_group, (Py_ssize_t) 2);


		if (section==NULL || param==NULL || value==NULL){
			PyErr_SetString(PyExc_ValueError, 
			"When creating an OptionSet you must use a list-like object of (section, param, value) triplets");

		    Py_DECREF(section);
		    Py_DECREF(param);
		    Py_DECREF(value);

		    des_optionset_free(optionset);

			return NULL;
		}
		
		//Convert the objects to strings - they may be numbers
		PyObject* section_string = PyObject_Str(section);
		PyObject*   param_string = PyObject_Str(param  );
		PyObject*   value_string = PyObject_Str(value  );

		//Release the original (possibly non-string) objects
	    Py_DECREF(section);
	    Py_DECREF(param);
	    Py_DECREF(value);

	    // Pull out the C strings from the objects
		char* section_ptr = PyString_AsString(section_string); 
		char*   param_ptr = PyString_AsString(param_string  );
		char*   value_ptr = PyString_AsString(value_string  ); 

		// Add the C strings to the optionset object
		des_optionset_add(optionset, section_ptr, param_ptr, value_ptr);

		//Release the string objects, as we are now done with them
	    Py_DECREF(section);
	    Py_DECREF(param);
	    Py_DECREF(value);

	    // and also release the triplet object
	    Py_DECREF(parameter_group);
	}

	Py_DECREF(iterator);

	return PyLong_FromLong((size_t) optionset);
}




static PyMethodDef 
desglue_methods[] = {
    {"create_fits",  desglue_create_fits, METH_VARARGS, "Turn a python string into an allocated internal FITS pointer object."},
    {"read_fits",  desglue_read_fits, METH_VARARGS, "Read a fits internal file back into a stringIO."},
    {"rewrite_fits",  desglue_rewrite_fits, METH_VARARGS, "Overwrite FITS data with a file string data."},
    {"free_fits",  desglue_free_fits, METH_VARARGS, "Free a fits internal file."},
    {"create_options", desglue_create_optionset_from_iterator, METH_VARARGS, "Turn a sequence of (section, name, value) parameters into an options object"},
    {"get_option", desglue_get_option_from_set, METH_VARARGS, "Extract an parameter value from an option set pointer, the section name, and the param name."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initdesglue(void)
{
    (void) Py_InitModule("desglue", desglue_methods);
}




