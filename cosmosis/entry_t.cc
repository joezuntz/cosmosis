#include "entry.hh"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>

using cosmosis::Entry;
using cosmosis::complex_t;

void test_int()
{
  Entry e(1);
  assert(e.is_int());
  assert(not e.is_double());
  assert(not e.is_string());
  assert(not e.is_complex());
  assert(e.int_val() == 1);
  e.set_int_val(10);
  assert (e.int_val() == 10);
  try {
    assert(e.double_val() == 10.0);
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is_int());
  e.set_string_val("cow");
  assert(e.is_string());
  assert(e.string_val() == "cow");
}

void test_double()
{
  Entry e(2.5);
  assert(not e.is_int());
  assert(e.is_double());
  assert(not e.is_string());
  assert(not e.is_complex());
  assert(e.double_val() == 2.5);
  e.set_double_val(-1.5);
  assert (e.double_val() == -1.5);
  try {
    assert(e.string_val() == "cow");
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is_double());
  e.set_string_val("cow");
  assert(e.is_string());
  assert(e.string_val() == "cow");
}

void test_string()
{
  Entry e("this has spaces");
  assert(not e.is_int());
  assert(not e.is_double());
  assert(e.is_string());
  assert(not e.is_complex());
  assert(e.string_val() == "this has spaces");
  e.set_string_val("cow");
  assert (e.string_val() == "cow");
  try {
    assert(e.double_val() == 1.5);
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is_string());
  e.set_complex_val(complex_t(1.0, -3.5));
  assert(e.is_complex());
  assert(e.complex_val() == complex_t(1.0, -3.5));
}

void test_complex()
{
  Entry e(complex_t(1.5, 2.5));
  assert(not e.is_int());
  assert(not e.is_double());
  assert(not e.is_string());
  assert(e.is_complex());
  assert(e.complex_val() == complex_t(1.5, 2.5));
  e.set_complex_val(complex_t(-2.5, 10.0));
  assert (e.complex_val() == complex_t(-2.5, 10.0));
  try {
    assert(e.int_val() == 1);
    assert(0 == "failed throw exception");
  }
  catch ( Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is_complex());
  e.set_int_val(-5);
  assert(e.is_int());
  assert(e.int_val() == -5);
}

template <class T, class F, class G, class H>
void test_vector(std::vector<T> const& vals,
                 F is_x_member,
                 G valuecheck_fun,
                 H set_x_member)
{
  // Create the entry with the specified values.
  Entry e(vals);
  // Make sure the type and value is what is expected.
  assert(std::mem_fn(is_x_member)(e));
  assert(valuecheck_fun(e, vals));
  // Now make the value be an non-vector but memory-managed type, so we
  // can observe the switching, and test for leaking memory.
  e.set_string_val("tomato soup");
  assert(e.is_string());
  // Now reverse the value, to make sure we have a different sequence of
  // the right type to assign.
  std::vector<T> reversed(vals);
  std::reverse(reversed.begin(), reversed.end());
  assert(vals != reversed);
  // ... set the values
  try { std::mem_fn(set_x_member)(e,reversed); }
  catch (...) { assert(0 == "setting array value threw unexpected exception"); }
  // ... test the values
  assert(std::mem_fn(is_x_member)(e));
  assert(valuecheck_fun(e, reversed));
  assert(not valuecheck_fun(e, vals));
}

int main()
{
  test_int();
  test_double();
  test_string();
  test_complex();
  test_vector(std::vector<int>({-101, 20, 3}),
              &Entry::is_int_array,
              [](Entry const& e, std::vector<int> v) -> bool { return e.int_array() == v; },
              &Entry::set_int_array
              );

  test_vector(std::vector<double>({-101.5, 2.0, 3.25, 1.875}),
              &Entry::is_double_array,
              [](Entry const& e, std::vector<double> v) -> bool { return e.double_array() == v; },
              &Entry::set_double_array
              );

  test_vector(std::vector<std::string>({"cow", "the dog"}),
              &Entry::is_string_array,
              [](Entry const& e, std::vector<std::string> v) -> bool { return e.string_array() == v; },
              &Entry::set_string_array
              );

  test_vector(std::vector<complex_t>({{-10.25,0.25}, {20.0, -3.0}}),
              &Entry::is_complex_array,
              [](Entry const& e, std::vector<complex_t> v) -> bool { return e.complex_array() == v; },
              &Entry::set_complex_array
              );

  Entry e("cats and dogs");
  std::cout << "size of Entry is: " << sizeof(Entry) << std::endl;
  std::cout << "size of e is:     " << sizeof(e) << std::endl;
}

