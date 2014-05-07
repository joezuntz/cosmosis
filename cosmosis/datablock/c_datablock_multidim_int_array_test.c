#include "c_datablock.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

int main()
{
  /* Array of two elements, each is an array of 3 ints. */
  int e_2_3[2][3] = { {1, 2, -1},
    {3, -5, 5 }
  };
  /* Array of three elements, each is an array of 2 ints. */
  int e_3_2[3][2] = { {1, 2}, { -1, 3}, { -5, 5} };
  /* First we have some tests that verify our compiler is treating
     multidimensional arrays according to the rules upon which we rely.
   */
  int sza = sizeof(e_2_3) / sizeof(int);
  assert(sza == 2 * 3);
  int szb = sizeof(e_3_2) / sizeof(int);
  assert(szb == 2 * 3);
  /* Make sure the two arrays give the same layout in memory */
  int* x = &(e_2_3[0][0]);
  int* y = &(e_3_2[0][0]);
  for (int i = 0; i != sza; ++i) { assert(x[i] == y[i]); }
  /* Insert a two-dimensional array into a datablock. */
  c_datablock* s = make_c_datablock();
  assert(s);
  const int expected_ndims = 2;
  int expected_extents[] = {2, 3};
  assert(c_datablock_put_int_array(s, "a", "e23",
                                   (int*)e_2_3,
                                   expected_ndims,
                                   expected_extents) == DBS_SUCCESS);
  assert(c_datablock_put_int_array(s, "a", "e23a",
                                   &(e_2_3[0][0]),
                                   expected_ndims,
                                   expected_extents) == DBS_SUCCESS);
  assert(c_datablock_has_value(s, "a", "e23") == true);
  assert(c_datablock_has_value(s, "a", "e23a"));
  /* Get what we just inserted */
  const int ndims = 2;
  int extents[ndims];
  assert(c_datablock_get_int_array_shape(s, "a", "e23",
                                         ndims, extents) == DBS_SUCCESS);
  assert(extents[0] == 2);
  assert(extents[1] == 3);
  int xx[extents[0]][extents[1]];
  assert(c_datablock_get_int_array(s, "a", "e23",
                                   (int*)xx, ndims, extents) == DBS_SUCCESS);
  for (int i = 0; i != 2; ++i)
    for (int j = 0; j != 3; ++j)
      { assert(xx[i][j] == e_2_3[i][j]); }
  assert(c_datablock_get_int_array(s, "a", "e23", &(xx[0][0]), ndims, extents) == DBS_SUCCESS);
  for (int i = 0; i != 2; ++i)
    for (int j = 0; j != 3; ++j)
      { assert(xx[i][j] == e_2_3[i][j]); }
  // Test a 4-dimensional array.
  int a4d[4][3][2][5];
  for (int i = 0; i != 4; ++i)
    for (int j = 0; j != 3; ++j)
      for (int k = 0; k != 2; ++k)
        for (int l = 0; l != 5; ++l)
          { a4d[i][j][k][l] = (i + 1) * (j + 2) * (k + 3) * (l + 4); }
  assert(! c_datablock_has_value(s, "a", "a4d"));
  int a4d_extents[] = {4, 3, 2, 5};
  assert(c_datablock_put_int_array(s, "a", "a4d", (int*)a4d, 4, a4d_extents)
         == DBS_SUCCESS);
  int out_extents[4];
  assert(c_datablock_get_int_array_shape(s, "a", "a4d", 4, &out_extents[0])
         == DBS_SUCCESS);
  for (int i = 0; i != 4; ++i) { assert(out_extents[i] == a4d_extents[i]); }
  int out4d[out_extents[0]][out_extents[1]][out_extents[2]][out_extents[3]];
  assert(c_datablock_get_int_array(s, "a", "a4d", &out4d[0][0][0][0], 3, out_extents) == DBS_NDIM_MISMATCH);
  assert(c_datablock_get_int_array(s, "a", "a4d", &out4d[0][0][0][0], 5, out_extents) == DBS_NDIM_MISMATCH);
  int wrong_extents[] = {4, 3, 2, 4};
  assert(c_datablock_get_int_array(s, "a", "a4d", &out4d[0][0][0][0], 4, wrong_extents) == DBS_EXTENTS_MISMATCH);
  assert(c_datablock_get_int_array(s, "a", "a4d", &out4d[0][0][0][0], 4, out_extents) == DBS_SUCCESS);
  for (int i = 0; i != 4; ++i)
    for (int j = 0; j != 3; ++j)
      for (int k = 0; k != 2; ++k)
        for (int l = 0; l != 5; ++l)
          { assert(out4d[i][j][k][l] == (i + 1) * (j + 2) * (k + 3) * (l + 4)); }
  destroy_c_datablock(s);
}
