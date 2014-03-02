#include <cassert>
#include <string>
#include <vector>

#include "section.hh"
#include "entry.hh"

using cosmosis::Section;
using cosmosis::complex_t;
using std::vector;
using std::string;

template <class T>
void test_type(T && x, T && y)
{
  Section s;
  assert(s.put_val("a", x) == DBS_SUCCESS);
  assert(s.put_val("a", y) == DBS_NAME_ALREADY_EXISTS);
  T result;
  assert(s.get_val("a", result) == DBS_SUCCESS);
  assert(result == x);
  assert(s.replace_val("a", y) == DBS_SUCCESS);
  assert(s.get_val("a", result) == DBS_SUCCESS);
  assert(result == y);
  assert(not s.has_value<T>("b"));
  assert(s.replace_val("b", x) == DBS_NAME_NOT_FOUND);
  assert(s.has_value<T>("a"));
  assert(s.has_name("a"));
  assert(not s.has_name("b"));
}

void test_crossing_types()
{
  Section s;
  assert(s.put_val("a", 1) == DBS_SUCCESS);
  assert(s.put_val("a", 2.0) == DBS_NAME_ALREADY_EXISTS);
  assert(s.replace_val("a", vector<string>()) == DBS_SUCCESS);
  assert(s.has_value<vector<string>>("a"));
}

int main()
{
  test_type(10, 101);
  test_type(-2.5, 1.0e17);
  test_type<std::string>("cow", "dog");
  test_type(complex_t(1.0, 2.0), complex_t(-5.5, 2e-12));
  test_type(vector<int>({1,2,3}), vector<int>({2,3,4,5}));
  test_type(vector<double>(), vector<double>({2.5,3.5,4.5,5.5}));
  test_type(vector<string>({"cow"}), vector<string>({"a","","cd"}));
  test_type(vector<complex_t>({1.5, 2.25}), vector<complex_t>());

  test_crossing_types();
}
