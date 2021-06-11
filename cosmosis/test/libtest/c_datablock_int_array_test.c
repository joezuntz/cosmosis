#include "c_datablock.h"
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#define TEST_ARRAY(length, val, expected) \
  for (int i = 0; i != length; ++i) assert(val[i] == expected[i])

int main()
{
  c_datablock* s = make_c_datablock();
  assert(s);
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
