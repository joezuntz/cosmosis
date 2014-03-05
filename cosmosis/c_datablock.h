#ifndef COSMOSIS_C_DATABLOCK_H
#define COSMOSIS_C_DATABLOCK_H

#include "datablock_status.h"
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
  The type c_datablock represents groups of named values.
  Values can be any of the following types:
      int, double, char array ("string"), double _Complex
      int array, double array, char array array, double _Complex array.

  Groups of parameters are organized into named 'sections'.

 */

  typedef void c_datablock;
  c_datablock* make_c_datablock(void);

  /*
    Return DBS_SUCCESS if the datablock has a section with the given
    name, and an error status otherwise.
   */
  DATABLOCK_STATUS c_datablock_has_section(c_datablock const* s, const char* name);

  /*
    Return the number of sections contained in the datablock. If s is
    null, return -1.
   */
  int c_datablock_num_sections(c_datablock const* s);

/*
  bool c_datablock_has_value(c_datablock const* s, const char* section, const char* name);

  DATABLOCK_STATUS c_datablock_get_section_name(..., int isection);
*/

DATABLOCK_STATUS destroy_c_datablock(c_datablock* s);


/*

  The c_datablock_get_T functions return DBS_SUCCESS if a value of
  type T with the given name is found in the given section, and an
  error status otherwise. No conversions of type are done.

  If the return status is not DBS_SUCCESS, the value written into 'val' is not
  defined.
*/

DATABLOCK_STATUS
c_datablock_get_int(c_datablock const* s, const char* section, const char* name, int* val);

DATABLOCK_STATUS
c_datablock_get_double(c_datablock const* s, const char* section, const char* name, double* val);

DATABLOCK_STATUS
c_datablock_get_complex(c_datablock const* s, const char* section, const char* name, double _Complex* val);

DATABLOCK_STATUS
c_datablock_get_string(c_datablock const* s, const char* section, const char* name, char** val);


/* Only scalars have default in the C and Fortran interfaces. */
DATABLOCK_STATUS
c_datablock_get_double_default(c_datablock const* s,
			       const char* section,
			       const char* name,
			       double* val,
			       double dflt);


/*
  Return 0 if the put worked, and nonzero to indicate failure.
  1: name already exists
  2: memory allocation failure
*/
DATABLOCK_STATUS
c_datablock_put_int(c_datablock* s, const char* section, const char* name, int val);

DATABLOCK_STATUS
c_datablock_put_double(c_datablock* s, const char* section, const char* name, double val);

DATABLOCK_STATUS
c_datablock_put_complex(c_datablock* s, const char* section, const char* name, double _Complex val);

DATABLOCK_STATUS
c_datablock_put_string(c_datablock* s, const char* section, const char* name, const char* val);

/*
  Return 0 if the put worked, and nonzero to indicate failure.
  1: name does not already exist.
  2: memory allocation failure.
  3: replace of wrong type.
*/
DATABLOCK_STATUS
c_datablock_replace_int(c_datablock* s, const char* section, const char* name, int val);

DATABLOCK_STATUS
c_datablock_replace_double(c_datablock* s, const char* section, const char* name, double val);

DATABLOCK_STATUS
c_datablock_replace_complex(c_datablock* s, const char* section, const char* name, double _Complex val);

DATABLOCK_STATUS
c_datablock_replace_string(c_datablock* s, const char* section, const char* name, const char* val);

#if 0
/* Return 0 if the put worked, and nonzero to indicate failure */
DATABLOCK_STATUS
c_datablock_get_double_array_1d(c_datablock const* s,
				const char* section,
				const char* name,
				double** array,
				int* size);

DATABLOCK_STATUS
c_datablock_get_double_array_1d_preallocated(c_datablock const* s,
					     const char* section,
					     const char* name,
					     double* array,
					     int* size,
					     int maxsize);

DATABLOCK_STATUS
c_datablock_put_double_array_1d(c_datablock* s,
				const char* section,
				const char* name,
				double* array,
				int sz);
#endif // if 0

#ifdef __cplusplus
}
#endif

#endif
