#ifndef COSMOSIS_DATABLOCK_TYPES_H
#define COSMOSIS_DATABLOCK_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/*
    datablock_type_t enumerates the varioust things that
    can be stored in the data data block
 */

typedef enum {
  DBT_INT,
  DBT_DOUBLE,
  DBT_COMPLEX,
  DBT_STRING,
  DBT_INT1D,
  DBT_DOUBLE1D,
  DBT_COMPLEX1D,
  DBT_STRING1D,
  DBT_INT2D,
  DBT_DOUBLE2D,
  DBT_COMPLEX2D,
  DBT_STRING2D,
  DBT_BOOL,
  DBT_UNKNOWN,
} datablock_type_t;



#ifdef __cplusplus
}
#endif

#endif
