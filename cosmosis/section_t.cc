#include <cassert>
#include <string>
#include <vector>

#include "section.hh"
#include "entry.hh"

using cosmosis::Section;
using cosmosis::complex_t;
using std::vector;
using std::string;

void test_puts()
{
  Section s1;

  assert(s1.put_val("i1", -10) == DBS_SUCCESS);
  assert(s1.put_val("d1", 2.5) == DBS_SUCCESS);
  assert(s1.put_val("s1", "cow says moo") == DBS_SUCCESS);
  assert(s1.put_val("z1", complex_t(-1.5, 3.5)) == DBS_SUCCESS);
  assert(s1.put_val("vi1", vector<int>({1, 2, -10})) == DBS_SUCCESS);
  assert(s1.put_val("vd1", vector<double>({2.5, -10.5})) == DBS_SUCCESS);
  assert(s1.put_val("vs1", vector<string>({"cow says moo", "pig says oink"})) == DBS_SUCCESS);
  assert(s1.put_val("vz1", vector<complex_t>({complex_t(-1.5, 3.5)})) == DBS_SUCCESS);

  for (auto name : { "i1", "d1", "s1", "z1", "vi1", "vd1", "vs1", "vz1" })
    assert(s1.put_val(name, "cow says moo") == DBS_NAME_ALREADY_EXISTS);
}

void test_replace()
{
  Section s;
  assert(s.replace_val("i", 21) == DBS_NAME_NOT_FOUND);

  assert(s.put_val("i", -10) == DBS_SUCCESS);
  assert(s.replace_val("i", 21) == DBS_SUCCESS);
  int i = 0;
  assert(s.get_val("i", i) == DBS_SUCCESS);
  assert(i == 21);
}

int main()
{
  test_puts();
  test_replace();
}
