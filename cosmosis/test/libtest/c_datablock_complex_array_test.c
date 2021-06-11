#include "c_datablock.h"
#include <assert.h>
#include <stdlib.h>
#include <float.h>

#define TEST_ARRAY(length, val, expected) \
  for (int i = 0; i != length; ++i) assert(val[i] == expected[i])

int main()
{
  c_datablock* s = make_c_datablock();
  assert(s);
  double _Complex expected[] = {1.25 + 1.25 * _Complex_I,
                                2.0 - 5.5 * _Complex_I,
                                3.5 * _Complex_I
                               };
  int sz = sizeof(expected) / sizeof(double _Complex);
  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_complex_array_1d(s, "x", "cow", expected, sz) == DBS_SUCCESS);
  double _Complex* val = NULL;
  int length;
  assert(c_datablock_get_complex_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);
  assert(c_datablock_get_array_length(s, "x", "cow") == sz);
  /* Get with preallocated buffer should return the right value. */
  const int big = 100;
  double _Complex buffer[big];
  assert(c_datablock_get_complex_array_1d_preallocated(s, "x", "cow", buffer, &length, big) == DBS_SUCCESS);
  /* Get with a too-small buffer should fail, and leave the buffer
     untouched. The returned value of length will say how big the
     buffer needs to be. */
  const int small = 1;
  double _Complex too_small[small];
  too_small[0] = DBL_MIN + DBL_MIN * _Complex_I;
  length = 0;
  assert(c_datablock_get_complex_array_1d_preallocated(s, "x", "cow", too_small, &length, small) ==
         DBS_SIZE_INSUFFICIENT);
  assert(too_small[0] == DBL_MIN + DBL_MIN * _Complex_I);
  assert(length == sz);
  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_complex_array_1d(s, "y", "cow", expected, sz) == DBS_SUCCESS);
  val = NULL;
  assert(c_datablock_get_complex_array_1d(s, "y", "cow", &val, &length) == DBS_SUCCESS);
  TEST_ARRAY(length, val, expected);
  free(val);
  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  double _Complex another_array[] = { 2.5,
                                      3.75 - 3.0 * _Complex_I,
                                      4.0 * _Complex_I,
                                      5.5 + 2.5 * _Complex_I
                                    };
  const int another_sz = sizeof(another_array) / sizeof(double _Complex);
  assert(c_datablock_put_complex_array_1d(s, "x", "cow", another_array, another_sz) == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  length = -1;
  assert(c_datablock_get_complex_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(sz != another_sz);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);
  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  length = -1;
  assert(c_datablock_get_complex_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(sz != another_sz);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);
  /* Replacement of an existing value with one of the same type should
     save the right value. */
  double _Complex new_expected[] = { 10.0 * _Complex_I,
                                     20.5 + 1.5 * _Complex_I,
                                     20.0 - 3.5 * _Complex_I,
                                     10.25 - 2.0 * _Complex_I,
                                     20.5 + 100 * _Complex_I,
                                     10.25 + 1.5 * _Complex_I
                                   };
  const int new_sz = sizeof(new_expected) / sizeof(double _Complex);
  assert(c_datablock_replace_complex_array_1d(s, "x", "cow", new_expected, new_sz) == DBS_SUCCESS);
  val = NULL;
  length = -1;
  assert(c_datablock_get_complex_array_1d(s, "x", "cow", &val, &length) == DBS_SUCCESS);
  assert(length == new_sz);
  TEST_ARRAY(length, val, new_expected);
  free(val);
  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_complex_array_1d(s, "x", "no such parameter", new_expected, new_sz) == DBS_NAME_NOT_FOUND);
  val = NULL;
  length = -1;
  assert(c_datablock_get_complex_array_1d(s, "x", "no such parameter", &val, &length) == DBS_NAME_NOT_FOUND);
  assert(val == NULL);
  assert(length == -1);
  /* no need to free val, because nothing was allocated. */
  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_complex(s, "x", "a double _Complex", 2.5) == DBS_SUCCESS);
  assert(c_datablock_replace_string(s, "x", "a double _Complex", "gurgle") == DBS_WRONG_VALUE_TYPE);
  double _Complex d;
  assert(c_datablock_get_complex(s, "x", "a double _Complex", &d) == DBS_SUCCESS);
  assert(d == 2.5);
  destroy_c_datablock(s);
}
