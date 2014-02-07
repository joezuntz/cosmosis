#include "sample.h"
#include <stdio.h>
#include <assert.h>

int main()
{
  double val;
  c_sample* s;
  s = make_c_sample();

  printf("Setting a double\n");
  c_sample_set_double(s, "cow", 2.5);

  c_sample_get_double(s, "cow", &val);
  printf("Got the value: %f\n", val);

  double x[] = {1,2,3};
  c_sample_put_double_array_1d(s, "pig", x, 3);

  double* y;
  int sz;
  c_sample_get_double_array_1d(s, "pig", &y, &sz);

  double z[4];
  int szz;
  c_sample_get_double_array_1d_preallocated(s, "pig", z, &szz, 4);
  assert(szz == 3);
  assert(z[0] == 1);
  assert(z[1] == 2);

  destroy_c_sample(s);
  return 0;
}
