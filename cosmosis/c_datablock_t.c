#include "datablock.h"
#include <stdio.h>
#include <assert.h>

int main()
{
  double val;
  c_datablock* s;
  s = make_c_datablock();
  assert(s);

  assert(c_datablock_put_double(s, "cow", 2.5) == 0);

  assert(c_datablock_get_double(s, "cow", &val) == 0);
  assert(val == 2.5);

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

  destroy_c_datablock(s);
  return 0;
}
