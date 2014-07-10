#include "c_datablock.h"


#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_sections()
{
  c_datablock* s = make_c_datablock();
  /* Null pointers should not cause crashes. */
  assert(c_datablock_has_section(NULL, NULL) == false);
  assert(c_datablock_has_section(s, NULL) == false);
  assert(c_datablock_num_sections(NULL) == -1);
  assert(c_datablock_has_value(NULL, NULL, NULL) == false);
  assert(c_datablock_has_value(s, NULL, NULL) == false);
  assert(c_datablock_has_value(s, "boo", NULL) == false);
  assert(c_datablock_has_section(s, "cow") == false);
  assert(c_datablock_num_sections(s) == 0);
  assert(c_datablock_get_array_length(NULL, NULL, NULL) == -1);
  assert(c_datablock_get_array_length(s, NULL, NULL) == -1);
  assert(c_datablock_get_array_length(s, "no such section", NULL) == -1);
  assert(c_datablock_get_value_name(NULL, "no such section", 0) == NULL);
  assert(c_datablock_get_value_name(s, NULL, 0) == NULL);
  assert(c_datablock_get_value_name(s, "no such section", 0) == NULL);
  assert(c_datablock_get_value_name_by_section_index(NULL, 0, 0) == NULL);
  assert(c_datablock_get_value_name_by_section_index(s, 0, 0) == NULL);
  assert(c_datablock_get_type(NULL, NULL, NULL, NULL) == DBS_DATABLOCK_NULL);
  assert(c_datablock_get_type(s, NULL, NULL, NULL) == DBS_SECTION_NULL);
  assert(c_datablock_get_type(s, "", NULL, NULL) == DBS_NAME_NULL);
  assert(c_datablock_get_type(s, "", "", NULL) == DBS_VALUE_NULL);
  /* Creating a parameter in a section must create the section. */
  assert(c_datablock_put_int(s, "s1", "a", 10) == DBS_SUCCESS);
  assert(c_datablock_has_section(s, "s1") == true);
  assert(c_datablock_num_sections(s) == 1);
  assert(c_datablock_has_value(s, "s1", "a") == true);
  assert(strcmp(c_datablock_get_value_name(s, "s1", 0), "a") == 0);
  assert(strcmp(c_datablock_get_value_name_by_section_index(s, 0, 0), "a") == 0);
  assert(c_datablock_has_value(s, "no such section", "a") == false);
  assert(c_datablock_has_value(s, "s1", "no such parameter") == false);
  /* Make a few more sections. */
  assert(c_datablock_put_int(s, "s2", "a", 10) == DBS_SUCCESS);
  assert(c_datablock_put_int(s, "s3", "a", 10) == DBS_SUCCESS);
  assert(c_datablock_put_double(s, "s4", "a", 10.5) == DBS_SUCCESS);
  const int expected_sections = 4;
  assert(c_datablock_num_sections(s) == expected_sections);
  /* Make sure all the section names are what we expect. We make the
     calls out-of-order intentionally.
   */
  assert(strcmp(c_datablock_get_section_name(s, 3), "s4") == 0);
  assert(strcmp(c_datablock_get_section_name(s, 0), "s1") == 0);
  assert(strcmp(c_datablock_get_section_name(s, 2), "s3") == 0);
  assert(strcmp(c_datablock_get_section_name(s, 1), "s2") == 0);
  assert(c_datablock_get_section_name(s, -1) == NULL);
  assert(c_datablock_get_section_name(s, 4) == NULL);
  destroy_c_datablock(s);
}

/*
  The header "test_c_datablock_scalars.h" is automatically generated.
 */

#include "test_c_datablock_scalars.h"

#define TEST_ARRAY(length, val, expected) \
  for (int i = 0; i != length; ++i) assert(val[i] == expected[i])

void test_array_int()
{
  c_datablock* s = make_c_datablock();
  assert(s);

  /* Test array length for non-array. */
  assert(c_datablock_put_int(s, "z", "i", 5) == DBS_SUCCESS);
  assert(c_datablock_get_array_length(s, "z", "i") == -1);

  int expected[] = {1, 2, 3};
  int sz = sizeof(expected) / sizeof(int);
  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_int_array_1d(s, "x", "cow", expected, sz) == DBS_SUCCESS);
  int* val = NULL;
  int length;
  assert(c_datablock_get_int_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);
  assert(c_datablock_get_array_length(s, "x", "cow") == sz);

  /* Get with preallocated buffer should return the right value. */
  const int big = 100;
  int buffer[big];
  assert(c_datablock_get_int_array_1d_preallocated(s, "x", "cow", buffer, &length, big) == DBS_SUCCESS);
  /* Get with a too-small buffer should fail, and leave the buffer
     untouched. The returned value of length will say how big the
     buffer needs to be. */
  const int small = 1;
  int too_small[small];
  too_small[0] = INT_MIN;
  length = 0;
  assert(c_datablock_get_int_array_1d_preallocated(s, "x", "cow", too_small, &length, small) ==
         DBS_SIZE_INSUFFICIENT);
  assert(too_small[0] == INT_MIN);
  assert(length == sz);
  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_int_array_1d(s, "y", "cow", expected, sz) == DBS_SUCCESS);
  val = NULL;
  assert(c_datablock_get_int_array_1d(s, "y", "cow", &val, &length) == DBS_SUCCESS);
  TEST_ARRAY(length, val, expected);
  free(val);
  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  int another_array[] = { 2, 3, 4, 5 };
  const int another_sz = sizeof(another_array) / sizeof(int);
  assert(c_datablock_put_int_array_1d(s, "x", "cow", another_array, another_sz) == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  length = -1;
  assert(c_datablock_get_int_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(sz != another_sz);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);
  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  length = -1;
  assert(c_datablock_get_int_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(sz != another_sz);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);
  /* Replacement of an existing value with one of the same type should
     save the right value. */
  int new_expected[] = { 10, 20, 20, 10, 20, 10 };
  const int new_sz = sizeof(new_expected) / sizeof(int);
  assert(c_datablock_replace_int_array_1d(s, "x", "cow", new_expected, new_sz) == DBS_SUCCESS);
  val = NULL;
  length = -1;
  assert(c_datablock_get_int_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(length == new_sz);
  TEST_ARRAY(length, val, new_expected);
  free(val);
  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_int_array_1d(s, "x", "no such parameter", new_expected, new_sz) == DBS_NAME_NOT_FOUND);
  val = NULL;
  length = -1;
  assert(c_datablock_get_int_array_1d(s, "x", "no such parameter", &val, &length) == DBS_NAME_NOT_FOUND);
  assert(val == NULL);
  assert(length == -1);
  /* no need to free val, because nothing was allocated. */
  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_double(s, "x", "a double", 2.5) == DBS_SUCCESS);
  assert(c_datablock_replace_string(s, "x", "a double", "gurgle") == DBS_WRONG_VALUE_TYPE);
  double d;
  assert(c_datablock_get_double(s, "x", "a double", &d) == DBS_SUCCESS);
  assert(d == 2.5);
  destroy_c_datablock(s);
}

void test_grid()
{
  c_datablock* s = make_c_datablock();
  DATABLOCK_STATUS status;
  int nx = 50;
  int ny = 100;
  double x[nx];
  double y[ny];
  double** z = allocate_2d_double(nx, ny);
  for (int i = 0; i < nx; i++)
    {
      x[i] = 100.0 * i;
    }
  for (int i = 0; i < ny; i++)
    {
      y[i] = -1.0 * i;
    }
  for (int i = 0; i < nx; i++)
    {
      for (int j = 0; j < ny; j++)
        {
          z[i][j] = x[i] + y[j];
        }
    }
  status = c_datablock_put_double_grid(s, "test",
                                       "X", nx, x,
                                       "Y", ny, y,
                                       "Z", z);
  assert(status == 0);
  int na, nb;
  double* a, *b, **c;
  status =  c_datablock_get_double_grid(s, "test",
                                        "X", &na, &a,
                                        "Y", &nb, &b,
                                        "Z", &c);
  assert(status == 0);
  for (int i = 0; i < nx; i++)
    {
      for (int j = 0; j < ny; j++)
        {
          assert(c[i][j] == x[i] + y[j]);
        }
    }
  free(a);
  free(b);
  deallocate_2d_double(&z, nx);
  deallocate_2d_double(&c, na);
  assert(status == 0);
  destroy_c_datablock(s);
}

void test_clone(){
  printf("In clone\n");
  c_datablock* s = make_c_datablock();

  assert(c_datablock_put_int(s, "A", "x", 3)==DBS_SUCCESS);

  int arr[] = {1, 2, 3};
  int sz = sizeof(arr) / sizeof(int);
  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_int_array_1d(s, "x", "cow", arr, sz) == DBS_SUCCESS);

  c_datablock * r = clone_c_datablock(s);
  destroy_c_datablock(s);
  s=NULL;

  int a=0;
  assert(c_datablock_get_int(r, "A", "x", &a)==DBS_SUCCESS);
  assert(a==3);

  int* val = NULL;
  int length;
  assert(c_datablock_get_int_array_1d(r, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(val[0]==arr[0] && val[1]==arr[1] && val[2]==arr[2]);

  destroy_c_datablock(r);

}

void test_c_copy(){
  printf("In c_copy\n");
  c_datablock* s = make_c_datablock();

  assert(c_datablock_put_int(s, "A", "x", 3)==DBS_SUCCESS);
  assert(c_datablock_num_sections(s)==1);
  assert(c_datablock_put_int(s, "B", "y", 3)==DBS_SUCCESS);
  assert(c_datablock_copy_section(s, "A", "B")==DBS_NAME_ALREADY_EXISTS);
  assert(c_datablock_copy_section(s, "A", "C")==DBS_SUCCESS);
  int x=-1;
  assert(c_datablock_get_int(s, "C", "x", &x)==DBS_SUCCESS);
  assert(x==3);
  destroy_c_datablock(s);

}

void test_ndim(){
  printf("In test_ndim\n");

{  c_datablock* s = make_c_datablock();
  DATABLOCK_STATUS status = 0;
  int x[4*4*4], y[5*5], z[17];
  int xsize[3], ysize[2], zsize[1];
  xsize[0] = 4; xsize[1] = 4; xsize[2] = 4;
  ysize[0] = 5; ysize[1] = 5;
  zsize[0] = 17;

  for (int i=0; i<4*4*4; i++) x[i] = 0;
  for (int i=0; i<5*5; i++) y[i] = 0;
  for (int i=0; i<17; i++) z[i] = 0;
  status = c_datablock_put_int_array(s, "sec", "x", x, 3, xsize);
  status = c_datablock_put_int_array(s, "sec", "y", y, 2, ysize);
  status = c_datablock_put_int_array(s, "sec", "z", z, 1, zsize);
  assert(status == 0);

  int ndim = 0;
  status |= c_datablock_get_array_ndim(s, "sec", "x", &ndim);
  assert(ndim == 3);
  status |= c_datablock_get_array_ndim(s, "sec", "y", &ndim);
  assert(ndim == 2);
  status |= c_datablock_get_array_ndim(s, "sec", "z", &ndim);
  assert(ndim == 1);
  assert(status == 0);

  destroy_c_datablock(s);
}
{
  c_datablock* s = make_c_datablock();
  DATABLOCK_STATUS status = 0;
  double x[4*4*4], y[5*5], z[17];
  int xsize[3], ysize[2], zsize[1];
  xsize[0] = 4; xsize[1] = 4; xsize[2] = 4;
  ysize[0] = 5; ysize[1] = 5;
  zsize[0] = 17;

  for (int i=0; i<4*4*4; i++) x[i] = 0;
  for (int i=0; i<5*5; i++) y[i] = 0;
  for (int i=0; i<17; i++) z[i] = 0;
  status = c_datablock_put_double_array(s, "sec", "x", x, 3, xsize);
  status = c_datablock_put_double_array(s, "sec", "y", y, 2, ysize);
  status = c_datablock_put_double_array(s, "sec", "z", z, 1, zsize);
  assert(status == 0);

  int ndim = 0;
  status |= c_datablock_get_array_ndim(s, "sec", "x", &ndim);
  assert(ndim == 3);
  status |= c_datablock_get_array_ndim(s, "sec", "y", &ndim);
  assert(ndim == 2);
  status |= c_datablock_get_array_ndim(s, "sec", "z", &ndim);
  assert(ndim == 1);
  assert(status == 0);

  destroy_c_datablock(s);

}

}


int main()
{
  test_sections();
  test_grid();
  test_scalar_int();
  test_scalar_double();
  test_scalar_string();
  test_scalar_complex();
  test_scalar_bool();
  test_array_int();
  test_ndim();
  test_c_copy();
  test_clone();
  return 0;
}
