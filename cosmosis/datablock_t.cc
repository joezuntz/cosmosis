#include "datablock.hh"
#include "entry.hh"

#include <cassert>
#include <string>
#include <vector>

using cosmosis::DataBlock;
using cosmosis::Section;
using cosmosis::complex_t;
using std::string;
using std::vector;

template <class T, class W>
void test(T const& x, T const& y, W const& wrong)
{
  DataBlock b;
  assert(not b.has_section("sect_a"));
  assert(b.put_val("sect_a", "param", x) == DBS_SUCCESS);
  assert(b.has_section("sect_a"));
  T val;
  assert(b.get_val("sect_a", "param", val) == DBS_SUCCESS);
  assert(val == x);
  assert(b.get_val("no such section", "param", val) == DBS_SECTION_NOT_FOUND);
  assert(b.get_val("sect_a", "no such parameter", val) == DBS_NAME_NOT_FOUND);

  assert(b.replace_val("sect_a", "no such parameter", val) == DBS_NAME_NOT_FOUND);
  assert(b.replace_val("no such section", "param", val) == DBS_SECTION_NOT_FOUND);
  assert(b.replace_val("sect_a", "param", wrong) == DBS_WRONG_VALUE_TYPE);
  assert(b.replace_val("sect_a", "param", y) == DBS_SUCCESS);
  assert(b.get_val("sect_a", "param", val) == DBS_SUCCESS);
  assert(val == y);
  try { b.view<T>("no such section", ""); assert(0 == "view<T> failed to throw exception\n");}
  catch (Section::BadSectionAccess const&) { }
  catch (...) { assert("view<T> threw the wrong type of exception\n"); }
  assert(b.view<T>("sect_a", "param") == y);
  try { b.view<T>("no such section", "param"); assert(0 == "view<T> failed to throw exception\n"); }
  catch (DataBlock::BadDataBlockAccess const&) { }
  catch (...) { assert("view<T> threw the wrong type of exception\n"); }
}

int main()
{
  test(100, -25, 2.5);
  test(2.5, -1.25e20, string("dog"));
  test(complex_t{10.5, 3.5}, complex_t{-2.5, -1.5}, 10);
  test(string("cow"), string("moose"), 20);
  test(vector<int>{1,2,3}, vector<int>{3,2,1}, vector<double>{1.5,25.});
  test(vector<double>{1,2,3}, vector<double>{3,2,1}, string("moo"));
  test(vector<complex_t>{{1,2},{2.5, 3}}, vector<complex_t>{{2,1}}, 100);
  test(vector<string>{"a","b","c"}, vector<string>{"dog", "cow"}, 1.5);
}
