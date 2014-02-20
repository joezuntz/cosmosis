#ifndef _H_PY_INTERNAL_FITS
#define _H_PY_INTERNAL_FITS
#include <Python.h>
#include <cStringIO.h>
#include "internal_fits.h"
#include "des_pipeline.h"
#include "des_options.h"

PyObject * convert_fits_to_stringio(internal_fits * F);
internal_fits * convert_string_to_fits(PyObject * string);

#endif
