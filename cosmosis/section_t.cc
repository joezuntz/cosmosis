#include <cassert>
#include <string>
#include <vector>

#include "section.hh"
#include "entry.hh"

using cosmosis::Section;
using cosmosis::complex_t;
using std::vector;
using std::string;

int main()
{
  Section s1;

  assert(s1.set_val("i1", -10) == DBS_SUCCESS);
  assert(s1.set_val("d1", 2.5) == DBS_SUCCESS);
  assert(s1.set_val("s1", "cow says moo") == DBS_SUCCESS);
  assert(s1.set_val("z1", complex_t(-1.5, 3.5)) == DBS_SUCCESS);

  assert(s1.set_val("vi1", vector<int>({1, 2, -10})) == DBS_SUCCESS);
  assert(s1.set_val("vd1", vector<double>({2.5, -10.5})) == DBS_SUCCESS);
  assert(s1.set_val("vs1", vector<string>({"cow says moo", "pig says oink"})) == DBS_SUCCESS);
  assert(s1.set_val("vz1", vector<complex_t>({complex_t(-1.5, 3.5)})) == DBS_SUCCESS);
}
