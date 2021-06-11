#include "datablock.hh"
#include "entry.hh"

#include <cassert>
#include <limits>
#include <string>
#include <vector>

using cosmosis::DataBlock;
using cosmosis::Section;
using cosmosis::complex_t;
using cosmosis::ndarray;
using std::string;
using std::vector;
using std::size_t;

template <class T>
void test_empty_datablock(T const& y)
{
  DataBlock b;
  assert(not b.has_section("sect_a"));
  T val;

  assert(b.get_val("no such section", "a", y, val) == DBS_SUCCESS);
  assert(val == y);
}

template <class T, class W>
void test(T const& x, T const& y, W const& wrong)
{
  test_empty_datablock(y);

  DataBlock b;
  T val;

  assert(b.put_val("SECT_A", "param", x) == DBS_SUCCESS);
  assert(b.has_section("sect_a"));
  assert(b.value_name(0,0)=="param");
  assert(b.value_name("sect_a",0)=="param");
  assert(b.has_section("Sect_a"));
  assert(b.has_section("Sect_A"));
  assert(b.get_val("sect_a", "parameter_will_be_defaulted", y, val) == DBS_SUCCESS);
  assert(val == y);

  assert(b.has_val("no such section", "x") == false);
  assert(b.has_val("sect_a", "parameter_will_be_defaulted") == true);
  assert(b.has_val("sect_a", "param") == true);

  assert(b.get_val("sect_a", "param", val) == DBS_SUCCESS);
  assert(val == x);
  assert(b.get_val("sect_A", "param", val) == DBS_SUCCESS);
  assert(val == x);
  assert(b.get_val("Sect_A", "PARAM", val) == DBS_SUCCESS);
  assert(val == x);

  assert(b.get_val("sect_a", "param", y, val) == DBS_SUCCESS);
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

void test_size()
{
  DataBlock b;
  assert(b.get_size("no such section", "cow") == -1);
  b.put_val("a", "x", 1);
  assert(b.get_size("a", "no such parameter") == -1);
  assert(b.get_size("a", "x") == -1);
  b.put_val("a", "y", vector<int>(100, 2));
  assert(b.get_size("a", "y") == 100);

  // The following would test what happens if we have an array-type
  // parameter that is too long... running it causes a memory exhaustion
  // on many machines, rather than exercising the functionality in the
  // DataBlock.
  //
  //  b.put_val("a","big", vector<int>(std::numeric_limits<int>::max()+1, 1.5));
  // assert(b.get_size("a", "big") == -2);
}

void test_sections()
{
  DataBlock b;
  assert(b.num_values("ints")==-1);
  b.put_val("ints", "a", 10);
  assert(b.num_values("ints")==1);
  b.put_val("doubles", "a", 2.5);
  b.put_val("strings", "a", string("cow says moo"));
  assert(b.num_sections() == 3);
  assert(b.section_name(0) == "doubles");
  assert(b.section_name(1) == "ints");
  assert(b.section_name(2) == "strings");
  b.put_val("doubles", "b", 3.5);  
  assert(b.num_values("doubles")==2);
  try { b.section_name(3); assert(0 == "section_name failed to throw required exception\n"); }
  catch (DataBlock::BadDataBlockAccess const&) { }
  catch (...) { assert(0 == "section_name threw the wrong type of exception\n"); }
}

void test_delete()
{
  DataBlock b;
  assert(b.put_val("ints", "a", 10)==DBS_SUCCESS);
  assert(b.num_sections() == 1);
  assert(b.has_val("ints", "a") == true);  
  assert(b.delete_section("ints")==DBS_SUCCESS);
  assert(b.num_sections() == 0);
  assert(b.has_val("ints", "a") == false);
  assert(b.delete_section("ints")==DBS_SECTION_NOT_FOUND);
  assert(b.put_val("ints", "a", 10)==DBS_SUCCESS);
  assert(b.has_val("ints", "a") == true);

}

void test_copy()
{
  DataBlock b;
  assert(b.put_val("A", "a", 10)==DBS_SUCCESS);
  assert(b.num_sections() == 1);
  assert(b.has_val("A", "a") == true);  
  assert(b.copy_section("A", "B")==DBS_SUCCESS);
  assert(b.num_sections() == 2);

  assert(b.has_val("B", "a") == true);  
  assert(b.has_val("A", "a") == true); 
  int a;
  assert(b.get_val("A", "a", a)==DBS_SUCCESS);
  assert(a==10);
  a=0;
  assert(b.get_val("B", "a", a)==DBS_SUCCESS);
  assert(a==10);
  assert(b.delete_section("A")==DBS_SUCCESS);
  assert(b.num_sections() == 1);
  a=0;
  assert(b.get_val("B", "a", a)==DBS_SUCCESS);
  assert(a==10);

}



void test_types()
{
  DataBlock b;
  b.put_val("ints", "a", 10);
  b.put_val("doubles", "a", 2.5);
  b.put_val("strings", "a", string("cow says moo"));
  b.put_val("complex", "a", complex_t{10.5, 3.5});
  b.put_val("int_vec", "a", vector<int>{3,2,1});
  b.put_val("double_vec", "a", vector<double>{3.,2.,1.});
  b.put_val("string_vec", "a", vector<string>{"3","2","1"});
  b.put_val("complex_vec", "a", vector<complex_t>{{1,2},{2.5, 3}});
  b.put_val("bools", "a", true);
  datablock_type_t t;
  assert (b.get_type("ints","a",t)==DBS_SUCCESS);
  assert(t==DBT_INT);
  assert (b.get_type("doubles","a",t)==DBS_SUCCESS);
  assert(t==DBT_DOUBLE);
  assert (b.get_type("complex","a",t)==DBS_SUCCESS);
  assert(t==DBT_COMPLEX);
  assert (b.get_type("strings","a",t)==DBS_SUCCESS);
  assert(t==DBT_STRING);
  assert (b.get_type("int_vec","a",t)==DBS_SUCCESS);
  assert(t==DBT_INT1D);
  assert (b.get_type("double_vec","a",t)==DBS_SUCCESS);
  assert(t==DBT_DOUBLE1D);
  assert (b.get_type("string_vec","a",t)==DBS_SUCCESS);
  assert(t==DBT_STRING1D);
  assert (b.get_type("complex_vec","a",t)==DBS_SUCCESS);
  assert(t==DBT_COMPLEX1D);
  assert (b.get_type("bools","a",t)==DBS_SUCCESS);
  assert(t==DBT_BOOL);
}

template <class T>
void test_multidim(T seed, vector<size_t> const& extents)
{
  DataBlock b;
  size_t num_elements = std::accumulate(begin(extents), end(extents), 1, 
                                        std::multiplies<size_t>());
  ndarray<T> val(std::vector<T>(num_elements, seed), extents);
  b.put_val("multi", "a", val);
}

int main()
{
  test(100, -25, 2.5);
  test(2.5, -1.25e20, string("dog"));
  test(complex_t{10.5, 3.5}, complex_t{-2.5, -1.5}, 10);
  test(string("cow"), string("moose"), 20);
  test(true, false, 2);
  test(vector<int>{1,2,3}, vector<int>{3,2,1}, vector<double>{1.5,25.});
  test(vector<double>{1,2,3}, vector<double>{3,2,1}, string("moo"));
  test(vector<complex_t>{{1,2},{2.5, 3}}, vector<complex_t>{{2,1}}, 100);
  test(vector<string>{"a","b","c"}, vector<string>{"dog", "cow"}, 1.5);
  test_size();
  test_sections();
  test_types();
  test_delete();
  test_copy();

  test_multidim(1.5, vector<size_t>{3,4,5});
}
