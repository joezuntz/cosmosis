#include "entry.hh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

using cosmosis::Entry;
using cosmosis::complex_t;
using cosmosis::vint_t;
using cosmosis::vdouble_t;
using cosmosis::vstring_t;
using cosmosis::vcomplex_t;
using std::vector;
using std::string;

void test_copy()
{
  using namespace std::rel_ops;

  Entry a(1);
  Entry a1(a);
  assert(a==a1);

  Entry b(2.5);
  Entry b1(b);
  assert(b==b1);
  assert(a!=b);

  Entry c("cow");
  Entry c1(c);
  assert(c==c1);
  assert(a!=c);
  assert(b!=c);

  Entry d(complex_t(1.5, 2.5));
  Entry d1(d);
  assert(d==d1);
  assert(a!=d);
  assert(b!=d);
  assert(c!=d);

  Entry e(vector<int>({1,2,10}));
  Entry e1(e);
  assert(e==e1);
  assert(e!=a);
  assert(e!=b);
  assert(e!=c);
  assert(e!=d);

  Entry f(vector<double>({1.25, -0.5}));
  Entry f1(f);
  assert(f==f1);
  assert(f!=a);
  assert(f!=b);
  assert(f!=c);
  assert(f!=d);
  assert(f!=e);
}

void test_mapusage()
{
  typedef std::map<string, Entry> map_t;
  map_t vals;
  vals.insert(map_t::value_type("cow", Entry("moo")));
  assert(vals.size()==1);
  assert(vals["cow"].val<string>() == "moo");
  assert(vals["cow"] == Entry("moo"));

  vals.insert(map_t::value_type("pi", Entry(4.0 * std::atan(1.0))));
  assert(vals.size()==2);
}

void test_int()
{
  Entry e(1);
  assert(e.is<int>());
  assert(not e.is<double>());
  assert(not e.is<string>());
  assert(not e.is<complex_t>());
  assert(e.val<int>() == 1);
  assert(e.size() == -1);
  e.set_val(10);
  assert (e.val<int>() == 10);
  try {
    assert(e.val<double>() == 10.0);
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is<int>());
  e.set_val("cow");
  assert(e.is<string>());
  assert(e.val<string>() == "cow");
}

void test_double()
{
  Entry e(2.5);
  assert(not e.is<int>());
  assert(e.is<double>());
  assert(not e.is<string>());
  assert(not e.is<complex_t>());
  assert(e.val<double>() == 2.5);
  assert(e.size() == -1);
  e.set_val(-1.5);
  assert(e.val<double>() == -1.5);
  try {
    assert(e.val<string>() == "cow");
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is<double>());
  e.set_val("cow");
  assert(e.is<string>());
  assert(e.val<string>() == "cow");
}

void test_string()
{
  Entry e("this has spaces");
  assert(not e.is<int>());
  assert(not e.is<double>());
  assert(e.is<string>());
  assert(not e.is<complex_t>());
  assert(e.val<string>() == "this has spaces");
  assert(e.size() == -1);
  e.set_val("cow");
  assert (e.val<string>() == "cow");
  try {
    assert(e.val<double>() == 1.5);
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is<string>());
  e.set_val(complex_t(1.0, -3.5));
  assert(e.is<complex_t>());
  assert(e.val<complex_t>() == complex_t(1.0, -3.5));
}

void test_complex()
{
  Entry e(complex_t(1.5, 2.5));
  assert(not e.is<int>());
  assert(not e.is<double>());
  assert(not e.is<string>());
  assert(e.is<complex_t>());
  assert(e.val<complex_t>() == complex_t(1.5, 2.5));
  assert(e.size() == -1);
  e.set_val(complex_t(-2.5, 10.0));
  assert (e.val<complex_t>() == complex_t(-2.5, 10.0));
  try {
    assert(e.val<int>() == 1);
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is<complex_t>());
  e.set_val(-5);
  assert(e.is<int>());
  assert(e.val<int>() == -5);
}

template <class T>
void test_vector(vector<T> const& vals)
{
  // Create the entry with the specified values.
  Entry e(vals);
  // Make sure the type and value is what is expected.
  assert(e.is<vector<T>>());
  assert(e.val<vector<T>>() == vals);
  assert(e.size() == vals.size());

  // Now make the value be an non-vector but memory-managed type, so we
  // can observe the switching, and test for leaking memory.
  e.set_val("tomato soup");
  assert(e.is<string>());

  // Now reverse the value, to make sure we have a different sequence of
  // the right type to assign.
  vector<T> reversed(vals);
  std::reverse(reversed.begin(), reversed.end());
  assert(vals != reversed);

  // ... set the values
  try { e.set_val(reversed); }
  catch (...) { assert(0 == "setting array value threw unexpected exception"); }
  // ... test the values
  assert(e.is<vector<T>>());
  assert(e.val<vector<T>>() == reversed);
  assert(e.val<vector<T>>() != vals);
}

int main()
{
  test_int();
  test_double();
  test_string();
  test_complex();
  test_vector(vector<int>({-101, 20, 3}));
  test_vector(vector<double>({-101.5, 2.0, 3.25, 1.875}));
  test_vector(vector<string>({"cow", "the dog"}));
  test_vector(vector<complex_t>({{-10.25,0.25}, {20.0, -3.0}}));
  test_copy();
  test_mapusage();

  Entry e("cats and dogs");
  std::cout << "size of Entry is: " << sizeof(Entry) << std::endl;
  std::cout << "size of e is:     " << sizeof(e) << std::endl;
}

