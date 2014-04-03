#ifndef COSMOSIS_C_DATABLOCK_H
#define COSMOSIS_C_DATABLOCK_H

#include "datablock_status.h"
#include "datablock_types.h"
#include <complex.h>
#include <stdbool.h>

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
    Return true if the datablock has a section with the given name, and
    false otherwise. If either 's' or 'name' is null, return false.
  */
  _Bool c_datablock_has_section(c_datablock const* s, const char* name);

  /*
    Return the number of sections contained in the datablock. If s is
    null, return -1.
  */
  int c_datablock_num_sections(c_datablock const* s);

  /*
    Return DBS_SUCCESS if the datablock has a value in the given section
    with the given name, and an error status otherwise. The associated
    value can be of any supported type.
  */
  _Bool c_datablock_has_value(c_datablock const* s, const char* section, const char* name);


  const char* c_datablock_get_value_name(c_datablock const* s, const char* section, int j);
  const char* c_datablock_get_value_name_by_section_index(c_datablock const* s, 
    int i, int j);


  /*
    If the section and name correspond to a value that is an array,
    return the length of the array. Otherwise return -1. If any of the
    arguments is NULL, return -1.
   */
  int c_datablock_get_array_length(c_datablock const* s, const char* section, const char* name);

  /*
    Get the number of values in the named section.
    If block and section are non-NULL and there is a section
    with the given name then return the number of values in that
    section.

    Otherwise return -1.

  */
  int c_datablock_num_values(
    c_datablock const* s, const char* section);


  /*
    Return the name of the i'th section of the datablock. Note that if a
    new section is added, the ordinal position of some or all of the
    named sections may change; this is because the sections are stored
    in sorted order for speed of lookup. The caller is not intended to
    free the returned pointer; the datablock retains ownership of the
    memory buffer containing the string. A NULL pointer is returned if i
    is negative or out-of-range. Numbering of sections starts with 0.
  */
  const char* c_datablock_get_section_name(c_datablock const* s, int i);

  /*
    Deallocate all the resources associated with the given datablock.
    After this call, any use of that datablock will result in undefined
    behavior (most likely, a crash in the program).
   */
  DATABLOCK_STATUS destroy_c_datablock(c_datablock* s);


  DATABLOCK_STATUS c_datablock_get_type(c_datablock const* s, const char* section, const char* name, datablock_type_t * t);

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
  c_datablock_get_bool(c_datablock const* s, const char* section, const char* name, bool* val);

  DATABLOCK_STATUS
  c_datablock_get_double(c_datablock const* s, const char* section, const char* name, double* val);

  DATABLOCK_STATUS
  c_datablock_get_complex(c_datablock const* s, const char* section, const char* name, double _Complex* val);

  DATABLOCK_STATUS
  c_datablock_get_string(c_datablock const* s, const char* section, const char* name, char** val);

  /* Only scalars have default in the C and Fortran interfaces. */
  DATABLOCK_STATUS
  c_datablock_get_int_default(c_datablock const* s,
                              const char* section,
                              const char* name,
                              int def,
                              int* val);

  DATABLOCK_STATUS
  c_datablock_get_bool_default(c_datablock const* s,
                              const char* section,
                              const char* name,
                              bool def,
                              bool* val);

  DATABLOCK_STATUS
  c_datablock_get_double_default(c_datablock const* s,
                                 const char* section,
                                 const char* name,
                                 double def,
                                 double* val);

  DATABLOCK_STATUS
  c_datablock_get_string_default(c_datablock const* s,
                                 const char* section,
                                 const char* name,
                                 const char* def,
                                 char** val);

  DATABLOCK_STATUS
  c_datablock_get_complex_default(c_datablock const* s,
                                  const char* section,
                                  const char* name,
                                  double _Complex def,
                                  double _Complex* val);


  /*
    Return 0 if the put worked, and nonzero to indicate failure.
    1: name already exists
    2: memory allocation failure
  */
  DATABLOCK_STATUS
  c_datablock_put_int(c_datablock* s, const char* section, const char* name, int val);

  DATABLOCK_STATUS
  c_datablock_put_bool(c_datablock* s, const char* section, const char* name, bool val);

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
  c_datablock_replace_bool(c_datablock* s, const char* section, const char* name, bool val);

  DATABLOCK_STATUS
  c_datablock_replace_double(c_datablock* s, const char* section, const char* name, double val);

  DATABLOCK_STATUS
  c_datablock_replace_complex(c_datablock* s, const char* section, const char* name, double _Complex val);

  DATABLOCK_STATUS
  c_datablock_replace_string(c_datablock* s, const char* section, const char* name, const char* val);

  /*
    Returns DBS_SUCCESS on success, and and error status otherwise.
    c_datablock_get_int_array_1d allocates takes as input the address of
    an array-of-int (val), into which is written the address of a
    newly-allocated array-of-int; it also takes an address of an int
    (size) into which is written the size of the newly-allocated array.

    The user is responsible for disposing of the allocated memory (using
    'free') when it is no longer needed.
   */
  DATABLOCK_STATUS
  c_datablock_get_int_array_1d(c_datablock const* s,
                               const char* section,
                               const char* name,
                               int** val,
                               int* size);

 /*
    Returns DBS_SUCCESS on success, and and error status otherwise.
    c_datablock_get_int_array_1d_preallocated takes as input an
    already-allocated array-of-int (val), into which is copied the
    values contained in the array. The user must specify the size of the
    pre-allocated array (maxsize). The function also takes the address
    of an int (size) into which is written the number of elements
    written into the pre-allocated array.

    If the size of the pre-allocated array smaller than the size of the
    array in the datablock, then no values are written into 'array', and
    and error status is returned. The size of the array in the datablock
    is still written into 'size' in this case.
   */
  DATABLOCK_STATUS
  c_datablock_get_int_array_1d_preallocated(c_datablock const* s,
                                            const char* section,
                                            const char* name,
                                            int* array,
                                            int* size,
                                            int maxsize);

  DATABLOCK_STATUS
  c_datablock_put_int_array_1d(c_datablock* s,
                               const char* section,
                               const char* name,
                               int const*  val,
                               int sz);

  DATABLOCK_STATUS
  c_datablock_replace_int_array_1d(c_datablock* s,
                                   const char* section,
                                   const char* name,
                                   int const* val,
                                   int sz);

  DATABLOCK_STATUS
  c_datablock_get_double_array_1d(c_datablock const* s,
                               const char* section,
                               const char* name,
                               double** val,
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
                               double const*  val,
                               int sz);



DATABLOCK_STATUS  datablock_put_double_grid(
  const char * section, 
  const char * name_x, int n_x, double * x,  
  const char * name_y, int n_y, double * y, 
  double ** P);




#ifdef __cplusplus
}
#endif

#endif
