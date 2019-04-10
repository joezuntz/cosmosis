#ifndef COSMOSIS_DATABLOCK_TYPES_H
#define COSMOSIS_DATABLOCK_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/*
    datablock_type_t enumerates the various types of things that can
    be stored in the c_datablock.
 */

typedef enum
{
  DBT_INT,
  DBT_DOUBLE,
  DBT_COMPLEX,
  DBT_STRING,
  DBT_INT1D,
  DBT_DOUBLE1D,
  DBT_COMPLEX1D,
  DBT_STRING1D,
  DBT_BOOL,
  DBT_INTND,
  DBT_DOUBLEND,
  DBT_COMPLEXND,
  DBT_UNKNOWN,
} datablock_type_t;

#ifdef __cplusplus
}
#endif

#endif
