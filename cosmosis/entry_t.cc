#include "entry.hh"
#include <cassert>

using cosmosis::Entry;
using cosmosis::complex_t;

void test_int()
{
  Entry e(1);
  assert(e.is_int());
  assert(!e.is_double());
  assert(!e.is_string());
  assert(!e.is_complex());
  assert(e.int_val() == 1);
  e.set_int_val(10);
  assert (e.int_val() == 10);
  try {
    assert(e.double_val() == 10.0);
    assert(0 == "failed throw exception");
  }
  catch ( cosmosis::Entry::BadEntry const & ) { }
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
  assert(!e.is_int());
  assert(e.is_double());
  assert(!e.is_string());
  assert(!e.is_complex());
  assert(e.double_val() == 2.5);
  e.set_double_val(-1.5);
  assert (e.double_val() == -1.5);
  try {
    assert(e.string_val() == "cow");
    assert(0 == "failed throw exception");
  }
  catch ( cosmosis::Entry::BadEntry const & ) { }
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
  assert(!e.is_int());
  assert(!e.is_double());
  assert(e.is_string());
  assert(!e.is_complex());
  assert(e.string_val() == "this has spaces");
  e.set_string_val("cow");
  assert (e.string_val() == "cow");
  try {
    assert(e.double_val() == 1.5);
    assert(0 == "failed throw exception");
  }
  catch ( cosmosis::Entry::BadEntry const & ) { }
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
  assert(!e.is_int());
  assert(!e.is_double());
  assert(!e.is_string());
  assert(e.is_complex());
  assert(e.complex_val() == complex_t(1.5, 2.5));
  e.set_complex_val(complex_t(-2.5, 10.0));
  assert (e.complex_val() == complex_t(-2.5, 10.0));
  try {
    assert(e.int_val() == 1);
    assert(0 == "failed throw exception");
  }
  catch ( cosmosis::Entry::BadEntry const & ) { }
  catch (...) {
    assert(0 == "threw wrong kind of exception");
  }

  assert(e.is_complex());
  e.set_int_val(-5);
  assert(e.is_int());
  assert(e.int_val() == -5);
}

#include <iostream>

int main()
{
  test_int();
  test_double();
  test_string();
  test_complex();

  Entry e("cats and dogs");
  std::cout << "size of Entry is: " << sizeof(Entry) << std::endl;
  std::cout << "size of e is:     " << sizeof(e) << std::endl;
}

