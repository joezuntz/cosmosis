#include "c_datablock.h"
#include <stdio.h>
#include <assert.h>

void check_scalar_double()
{
  double val, expected;
  c_datablock* s;
  s = make_c_datablock();
  assert(s);

  expected = 3.5;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_double(s, "cow", expected) == 0);
  assert(c_datablock_get_double(s, "cow", &val) == 0);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_double(s, "cow", 10.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, "cow", &val) == 0);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, "cow", &val) == 0);
  assert(val == expected);

  destroy_c_datablock(s);
}

void check_scalar_int()
{
  int val, expected;
  c_datablock* s;
  s = make_c_datablock();
  assert(s);

  expected = -4;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_int(s, "cow", expected) == 0);
  assert(c_datablock_get_int(s, "cow", &val) == 0);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_int(s, "cow", 100) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, "cow", &val) == 0);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_double(s, "cow", 10.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, "cow", &val) == 0);
  assert(val == expected);

  destroy_c_datablock(s);
}


  /*
  double x[] = {1,2,3};
  c_datablock_put_double_array_1d(s, "pig", x, 3);
  */

  /*
  double* y;
  int sz;
  c_datablock_get_double_array_1d(s, "pig", &y, &sz);
  */

  /*
  double z[4];
  int szz;
  c_datablock_get_double_array_1d_preallocated(s, "pig", z, &szz, 4);
  assert(szz == 3);
  assert(z[0] == 1);
  assert(z[1] == 2);
  */


int main()
{
  check_scalar_double();
  check_scalar_int();
  return 0;
}
