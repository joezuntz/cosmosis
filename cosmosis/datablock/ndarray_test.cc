#include "ndarray.hh"
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

using cosmosis::ndarray;
using std::vector;

int
main()
{
  // Array of 3 elements, each an array of 2 doubles.
  double x[3][2] = {{1, 2}, {3, 4}, {5, 6}};
  assert(sizeof(x) / sizeof(double) == 3 * 2);

  // Test elementary construction, with different calling styles.
  const int ndim = 2;
  const int extents[] = {3, 2};
  ndarray<double> xx(&(x[0][0]), ndim, extents);
  ndarray<double> yy(reinterpret_cast<double*>(x), ndim, extents);
  assert(xx.ndims() == ndim);
  assert(xx == yy);

  // Assure that range-for loops work. They go over all elements in
  // linear order.
  vector<double> vals;
  for (double x : xx)
    vals.push_back(x);
  double* b = &x[0][0];
  assert(vals == vector<double>(b, b + 6));

  // Make sure size is reported correctly. Size is the number of array
  // elements stored.
  assert(xx.size() == 3 * 2);
  assert(yy.size() == 3 * 2);

  // Assure that subscripting of the ndarray works as we would expect.

  for (size_t i = 0; i != 3; ++i)
    for (size_t j = 0; j != 2; ++j)
      assert(xx(i, j) == x[i][j]);

  xx(1ul, 1ul) = 20;
  assert(xx(1ul, 1ul) == 20);

  double x3D[3][2][2] = {
    {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}};

  int extents3D[3] = {3, 2, 2};
  ndarray<double> xxx(&(x3D[0][0][0]), 3, extents3D);

  // Make sure use of the wrong number of indices causes an exception throw.
  try {
    xxx(1ul,2ul) = 2.5;
    assert("Failed to throw expected exception" == 0);
  } catch (cosmosis::NDArrayIndexException const& x) {
    //expected
  } catch (...) {
    assert("Caught wrong type of exception after bad ndarray indexing." == 0);
  }

  for (size_t i = 0; i != 3; ++i)
    for (size_t j = 0; j != 2; ++j)
      for (size_t k = 0; k != 2; ++k)
        assert(xxx(i, j, k) == x3D[i][j][k]);

  xxx(1ul, 1ul, 1ul) = 100;
  assert(xxx(1ul, 1ul, 1ul) == 100);
}
