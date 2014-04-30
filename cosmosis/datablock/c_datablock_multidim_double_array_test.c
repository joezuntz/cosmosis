#include "c_datablock.h"
#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

#define TEST_ARRAY(length, val, expected) \
  for (int i = 0; i != length; ++i) assert(val[i] == expected[i])

int main()
{
  /* Array of two elements, each is an array of 3 doubles. */
  double e_2_3[2][3] = { {1.25, 2.0, -0.5},
                         {3.5,-5.0, 5.5 } };

  /* Array of three elements, each is an array of 2 doubles. */
  double e_3_2[3][2] = { {1.25, 2.0}, {-0.5, 3.5}, {-5.0, 5.5} };

  /* First we have some tests that verify our compiler is treating
     multidimensional arrays according to the rules upon which we rely.
   */
  int sza = sizeof(e_2_3)/sizeof(double);
  assert(sza == 2*3);

  int szb = sizeof(e_3_2)/sizeof(double);
  assert(szb == 2*3);

  /* Make sure the two arrays give the same layout in memory */
  double* x = &(e_2_3[0][0]);
  double* y = &(e_3_2[0][0]);
  for (int i = 0; i != sza; ++i) assert( x[i] == y[i] );

  /* Insert a two-dimensional array into a datablock. */

  c_datablock* s = make_c_datablock();
  assert(s);

  const int expected_ndims = 2;
  int expected_extents[] = {2,3};
  assert(c_datablock_put_double_array(s, "a", "e23", (double*)e_2_3, expected_ndims, expected_extents) == DBS_SUCCESS);
  //assert(c_datablock_put_double_array(s, "a", "e23", &(e_2_3[0][0]), expected_ndims, expected_extents) == DBS_SUCCESS);
  assert(c_datablock_has_value(s, "a", "e23") == true);

  /* Get what we just inserted */
  const int ndims = 2;
  int extents[ndims];
  assert(c_datablock_get_double_array_shape(s, "a", "e23", ndims, extents) == DBS_SUCCESS);
  assert(extents[0] == 2);
  assert(extents[1] == 3);
  double xx[extents[0]][extents[1]];
  assert(c_datablock_get_double_array(s, "a", "e23", (double*)xx, ndims, extents) == DBS_SUCCESS);
  //assert(c_datablock_get_double_array(s, "a", "e23", &(xx[0][0]), ndims, extents) == DBS_SUCCESS);
  for (int i = 0; i != 2; ++i)
    for (int j = 0; j != 3; ++j)
      assert(xx[i][j] == e_2_3[i][j]);

  destroy_c_datablock(s);
}
