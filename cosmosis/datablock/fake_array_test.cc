#include <iostream>
#include <typeinfo>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cassert>
#include <memory>

#include "mdarraygen.hh"
#include "fakearray.hh"

using namespace std;

int main()
{
  using D1234 = MDArrayGen<double,1,2,3,4>;
  cout << "extent count for array dim 1,2,3,4: " << D1234::ndims << "\n";
  cout << "first extent value for array dim 1,2,3,4: " << D1234::extent << "\n";
  cout << "-----\n";
  assert(D1234::ndims == 4);
  assert(D1234::extent == 1);

  assert(D1234::inner_t::ndims == 3);
  assert(D1234::inner_t::extent == 2);

  assert(D1234::inner_t::inner_t::ndims == 2);
  assert(D1234::inner_t::inner_t::extent == 3);

  assert(D1234::inner_t::inner_t::inner_t::ndims == 1);
  assert(D1234::inner_t::inner_t::inner_t::extent == 4);

  assert(D1234::inner_t::inner_t::inner_t::inner_t::ndims == 0);
  assert(D1234::inner_t::inner_t::inner_t::inner_t::extent == 0);

  // Make sure reference type works as expected.
  using F745 = MDArrayGen<float,7,4,5>;
  F745::type q;
  F745::reference q_ref(q);
  assert(q == q_ref);

  float (&rq)[7][4][5] = q; // hard-coded reference to data
  q[6][2][3]=1.75; // set element using real data
  assert( q[6][2][3] == rq[6][2][3]);

  cout << "sizeof(F745) = " << sizeof(q) << "\n";
  assert(sizeof(q) == sizeof(float)*7*5*4);
  cout << "byte count of F745 = " << F745::size_bytes << "\n";
  assert(F745::size_bytes == sizeof(q));
  cout << "element count of F745 = " << F745::size_elements << "\n";
  assert(F745::size_elements == 7*5*4);
  cout << "type name for F745: " << typeid(F745::type).name() << "\n";
  assert(typeid(F745::type) == typeid(q));
  cout << "type name for ref to F745: " << typeid(F745::reference).name() << "\n";
  assert(typeid(F745::reference) == typeid(rq));
  cout << "type name for ptr to F745: " << typeid(F745::pointer).name() << "\n";
  assert(typeid(F745::pointer) == typeid(&q));

  // Make sure all the values in the array are accessible as if the
  // FakeArray was a real array.
  using Fake4d = FakeArray<double, 8,6,5,4>;
  assert(Fake4d::size_bytes == 8*6*5*4*sizeof(double));
  // Make sure Fake4d::size_bytes is a constant expression.
  typedef std::array<char, Fake4d::size_bytes> this_should_compile;
  assert(sizeof(this_should_compile) == Fake4d::size_bytes);

  // Allocate buffer for storage of the array elements, and fill with
  // known data.
  size_t nelements = Fake4d::size_bytes/sizeof(double);
  // The unique_ptr will delete the buffer, so we don't have to.
  unique_ptr<double[]> buffer { new double[nelements] };
  double* data = buffer.get();
  int init = 1;
  for (double *b = data, *e = data+nelements; b!=e; ++b) *b = (init++)*1.5;
  
  // Put 4d array on top of buffer
  Fake4d f4(data);
  cout << "after f4" << endl;
  for (int i = 0; i != 7; ++i)
    for (int j = 0; j != 5; ++j)
      for (int k = 0; k != 4; ++k)
        for (int l = 0; l != 3; ++l)
          assert(f4.data[i][j][k][l] == f4[i][j][k][l]);
  
  f4[7][5][4][2] = 1.5;
  cout << "FakeArray: element 7,5,4,2 using cast should 1.5, value is " << f4[7][5][4][2] << "\n";
  
  // obtain reference to buffer as 4d array
  auto f4a = make_fake_array<double, 8,6,5,4>(data);
  cout << "make_fake_array: element 7,5,4,2 should be 1.5, value is " << f4a[7][5][4][2] << "\n";
  for (int i = 0; i != 7; ++i)
    for (int j = 0; j != 5; ++j)
      for (int k = 0; k != 4; ++k)
        for (int l = 0; l != 3; ++l)
          assert(f4[i][j][k][l] == f4a[i][j][k][l]);

  // Test extent checker.
  std::vector<size_t> ext = { 8,6,5,4 };
  std::vector<size_t> ext_wrong = { 9,6,5,4 };
  std::vector<size_t> other_wrong = { 8, 6 };      
  cout << "matching extents for 8,6,5,4 using 8,6,5,4 -> " << verifyExtents<8,6,5,4>(ext) << " " << "\n";
  assert((verifyExtents<8,6,5,4>(ext)));
  assert((!verifyExtents<8,6,5,4>(ext_wrong)));
  cout << "matching extents for 8,6,5,4 using 9,6,5,4 -> " << verifyExtents<8,6,5,4>(ext_wrong) << " " << "\n";
  assert((!verifyExtents<8,6,5,4>(other_wrong)));
}
