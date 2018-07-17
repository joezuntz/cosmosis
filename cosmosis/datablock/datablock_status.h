#ifndef COSMOSIS_DATABLOCK_STATUS_H
#define COSMOSIS_DATABLOCK_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
  DATABLOCK_STATUS is type of the status codes returned by the C
  datablock interface functions that return status codes. In all cases
  the user can rely upon 0 indicating success, and nonzero indicating
  some type of failure. The exact numeric values of codes indicating
  failure should not be relied upon; use the enumerators instead of
  integral values.
*/

typedef enum
{
  /* DBS_SUCCESS is returned to indicate success for most functions. */
  DBS_SUCCESS = 0,
  DBS_DATABLOCK_NULL,
  DBS_SECTION_NULL,
  DBS_SECTION_NOT_FOUND,
  DBS_NAME_NULL,
  DBS_NAME_NOT_FOUND,
  DBS_NAME_ALREADY_EXISTS,
  DBS_VALUE_NULL,
  DBS_WRONG_VALUE_TYPE,
  DBS_MEMORY_ALLOC_FAILURE,
  DBS_SIZE_NULL,
  DBS_SIZE_NONPOSITIVE,
  DBS_SIZE_INSUFFICIENT,
  DBS_NDIM_NONPOSITIVE,
  DBS_NDIM_OVERFLOW,
  DBS_NDIM_MISMATCH,
  DBS_EXTENTS_NULL,
  DBS_EXTENTS_MISMATCH,
  DBS_LOGIC_ERROR,
  /*
    DBS_USED_DEFAULT should never be returned by a user-facing function.
  */
  DBS_USED_DEFAULT,
} DATABLOCK_STATUS;

inline const char * datablock_status_str(DATABLOCK_STATUS dbs) {
  switch (dbs) {
    case DBS_SUCCESS:
      return "DBS_SUCCESS";
    case DBS_DATABLOCK_NULL:
      return "DBS_DATABLOCK_NULL";
    case DBS_SECTION_NULL:
      return "DBS_SECTION_NULL";
    case DBS_SECTION_NOT_FOUND:
      return "DBS_SECTION_NOT_FOUND";
    case DBS_NAME_NULL:
      return "DBS_NAME_NULL";
    case DBS_NAME_NOT_FOUND:
      return "DBS_NAME_NOT_FOUND";
    case DBS_NAME_ALREADY_EXISTS:
      return "DBS_NAME_ALREADY_EXISTS";
    case DBS_VALUE_NULL:
      return "DBS_VALUE_NULL";
    case DBS_WRONG_VALUE_TYPE:
      return "DBS_WRONG_VALUE_TYPE";
    case DBS_MEMORY_ALLOC_FAILURE:
      return "DBS_MEMORY_ALLOC_FAILURE";
    case DBS_SIZE_NULL:
      return "DBS_SIZE_NULL";
    case DBS_SIZE_NONPOSITIVE:
      return "DBS_SIZE_NONPOSITIVE";
    case DBS_SIZE_INSUFFICIENT:
      return "DBS_SIZE_INSUFFICIENT";
    case DBS_NDIM_NONPOSITIVE:
      return "DBS_NDIM_NONPOSITIVE";
    case DBS_NDIM_OVERFLOW:
      return "DBS_NDIM_OVERFLOW";
    case DBS_NDIM_MISMATCH:
      return "DBS_NDIM_MISMATCH";
    case DBS_EXTENTS_NULL:
      return "DBS_EXTENTS_NULL";
    case DBS_EXTENTS_MISMATCH:
      return "DBS_EXTENTS_MISMATCH";
    case DBS_LOGIC_ERROR:
      return "DBS_LOGIC_ERROR";
    case DBS_USED_DEFAULT:
      return "DBS_USED_DEFAULT";
  }
  return "INVALID DATA BLOCK STATUS";
}

#ifdef __cplusplus
}
#endif

#endif
