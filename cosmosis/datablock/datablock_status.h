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
  failure should not be relied upon.
 */

typedef enum {
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
  DBS_LOGIC_ERROR,
  DBS_NDIM_NONPOSITIVE,
  DBS_NDIM_OVERFLOW,
  DBS_NDIM_MISMATCH,
  DBS_EXTENTS_NULL,
  DBS_EXTENTS_MISMATCH,
  DBS_LOGIC_ERROR
  // This one is a little different - it should only be returned internall
  // by the Section object rather than by any user-facing code
  DBS_USED_DEFAULT, 
} DATABLOCK_STATUS;

#ifdef __cplusplus
}
#endif

#endif
