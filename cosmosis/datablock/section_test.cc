#include <cassert>
#include <string>
#include <vector>

#include "section.hh"
#include "entry.hh"
#include "datablock_types.h"

using cosmosis::Entry;
using cosmosis::Section;
using cosmosis::complex_t;
using std::vector;
using std::string;

template <class T>
void test_type(T && x, T && y)
{
  Section s;
  T result;

  assert(s.get_val("no such parameter", x, result) == DBS_USED_DEFAULT);
  assert(result == x);

  assert(s.put_val("a", x) == DBS_SUCCESS);
  assert(s.put_val("a", y) == DBS_NAME_ALREADY_EXISTS);
  assert(s.value_name(0)=="a");
  assert(s.get_val("a", result) == DBS_SUCCESS);
  assert(result == x);
  assert(s.get_val("a", y, result) == DBS_SUCCESS);
  assert(result == x);
  assert(s.get_val("no such parameter", result) == DBS_NAME_NOT_FOUND);
  try { s.view<T>("no such parameter"); assert(0 =="view<T> failed to throw an exception"); }
  catch (Section::BadSectionAccess const&) { }
  catch (...) { assert(0 =="view<T> threw wrong type of exception"); }
  assert(result == x);
  assert(s.replace_val("a", y) == DBS_SUCCESS);
  assert(s.get_val("a", result) == DBS_SUCCESS);
  assert(result == y);
  assert(s.view<T>("a") == y);
  assert(not s.has_value<T>("b"));
  assert(s.replace_val("b", x) == DBS_NAME_NOT_FOUND);
  assert(s.has_value<T>("a"));
  assert(s.has_val("a"));
  assert(not s.has_val("b"));
}

void test_crossing_types()
{
  Section s;
  assert(s.put_val("a", 1) == DBS_SUCCESS);
  assert(s.put_val("a", 2.0) == DBS_NAME_ALREADY_EXISTS);
  assert(s.replace_val("a", vector<string>()) == DBS_WRONG_VALUE_TYPE);
  assert(s.has_value<int>("a"));
}

void test_section_size()
{
  Section s;
  assert(s.number_values() == 0);
  assert(s.put_val("a", 1) == DBS_SUCCESS);
  assert(s.number_values() == 1);
  assert(s.put_val("b", 2.5) == DBS_SUCCESS);
  assert(s.number_values() == 2);
  assert(s.replace_val("b",3.4)==DBS_SUCCESS);
  assert(s.number_values() == 2);
}

void test_size()
{
  Section s;
  assert(s.put_val("a", 1) == DBS_SUCCESS);
  assert(s.get_size("a") == -1);

  assert(s.put_val("b", 2.5) == DBS_SUCCESS);
  assert(s.get_size("b") == -1);

  assert(s.put_val("c", "cow") == DBS_SUCCESS);
  assert(s.get_size("c") == -1);

  assert(s.put_val("d", complex_t(1.5, 2.5)) == DBS_SUCCESS);
  assert(s.get_size("d") == -1);

  assert(s.put_val("e", vector<int>(102, 1)) == DBS_SUCCESS);
  assert(s.get_size("e") == 102);

  assert(s.put_val("f", vector<double>(1024*1024, -0.5)) == DBS_SUCCESS);
  assert(s.get_size("f") == 1024*1024);

  assert(s.put_val("g", vector<complex_t>(103, complex_t(1.5, 3.5))) == DBS_SUCCESS);
  assert(s.get_size("g") == 103);

  assert(s.put_val("h", vector<string>(99, "dog")) == DBS_SUCCESS);
  assert(s.get_size("h") == 99);
}



void test_type_finding()
{
  Section s;
  datablock_type_t t;
  assert(s.put_val("a", 1) == DBS_SUCCESS);
  assert(s.get_type("a",t) == DBS_SUCCESS);
  assert(t==DBT_INT);

  assert(s.put_val("b", 2.5) == DBS_SUCCESS);
  assert(s.get_type("b",t) == DBS_SUCCESS);
  assert(t==DBT_DOUBLE);

  assert(s.put_val("c", "cow") == DBS_SUCCESS);
  assert(s.get_type("c",t) == DBS_SUCCESS);
  assert(t==DBT_STRING);

  assert(s.put_val("d", complex_t(1.5, 2.5)) == DBS_SUCCESS);
  assert(s.get_type("d",t) == DBS_SUCCESS);
  assert(t==DBT_COMPLEX);

  assert(s.put_val("e", vector<int>(102, 1)) == DBS_SUCCESS);
  assert(s.get_type("e",t) == DBS_SUCCESS);
  assert(t==DBT_INT1D);

  assert(s.put_val("f", vector<double>(1024*1024, -0.5)) == DBS_SUCCESS);
  assert(s.get_type("f",t) == DBS_SUCCESS);
  assert(t==DBT_DOUBLE1D);

  assert(s.put_val("g", vector<complex_t>(103, complex_t(1.5, 3.5))) == DBS_SUCCESS);
  assert(s.get_type("g",t) == DBS_SUCCESS);
  assert(t==DBT_COMPLEX1D);

  assert(s.put_val("h", vector<string>(99, "dog")) == DBS_SUCCESS);
  assert(s.get_type("h",t) == DBS_SUCCESS);
  assert(t==DBT_STRING1D);
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
  test_size();
  test_section_size();
  test_type_finding();
  
  Section s;
  Section s2(s);
  Section s3;
  s3 = s;
  //assert(s2 == s3);
}
