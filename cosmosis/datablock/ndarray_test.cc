#include "ndarray.hh"
#include <cassert>
#include <vector>
#include <cstdio>

using cosmosis::ndarray;
using std::vector;

int main()
{
  // Array of 3 elements, each an array of 2 doubles.
  double x[3][2] = { {1,2},{3,4},{5,6} };
  assert(sizeof(x)/sizeof(double) == 3*2);

  // Test elementary construction, with different calling styles. 
  const int ndim = 2;
  const int extents[2] = {3, 2};
  ndarray<double> xx(&(x[0][0]), ndim, &extents[0]);
  ndarray<double> yy(reinterpret_cast<double*>(x), ndim, &extents[0]);
  assert(xx.ndims() == ndim);
  assert(xx==yy);

  // Assure that range-for loops work. They go over all elements in
  // linear order.
  vector<double> vals;
  for (double x : xx) vals.push_back(x);
  double* b = &x[0][0];
  assert(vals == vector<double>(b, b+6));

  // Make sure size is reported correctly. Size is the number of array
  // elements stored.
  assert(xx.size() == 3*2);
  assert(yy.size() == 3*2);



  // The code below is one example of what we would like the C++
  // interface of the ndarray to support. This is not yet working, and
  // the real solution may end up looking different from this.
  //
  // Assure that subscripting of the ndarray works as we would expect.

  for (size_t i = 0; i != 3; ++i)
    for (size_t j = 0; j != 2; ++j)
      assert(xx(i,j) == x[i][j]);

  xx(1,1) = 20;
  assert(xx(1,1)==20);

  double x3D[3][2][2] = {
      {{0,1},{2,3}},{{4,5},{6,7}},{{8,9},{10,11}}
    };

  int extents3D[3]  = {3,2,2};
  ndarray<double> xxx(&(x3D[0][0][0]), 3, &extents3D[0]);


  for (size_t i = 0; i != 3; ++i)
    for (size_t j = 0; j != 2; ++j)
      for (size_t k = 0; k != 2; ++k)
        assert(xxx(i,j,k) == x3D[i][j][k]);

  xxx(1,1,1) = 100;
  assert(xxx(1,1,1) == 100);

  // int xxxx[16];

  // for (int i=0; i<16; i++){
  //   xxxx[i] = i;
  // }

  // int dims4[4] = {2,2,2,2};
  // ndarray<int> x4D(xxxx, 4, dims4);
  // for (size_t i=0; i<2; i++){
  //   for (size_t j=0; j<2; j++){
  //     for (size_t k=0; k<2; k++){
  //       for (size_t l=0; l<2; l++){
  //         printf("X[%ld,%ld,%ld,%ld] = %d\n",i,j,k,l,x4D(i,j,k,l));
  //       }
  //     }
  //   }
  // }

}
