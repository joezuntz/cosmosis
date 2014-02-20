#include "section.hh"
#include "entry.hh"

using cosmosis::Section;
using cosmosis::complex_t;

int main()
{
  Section s1;
  s1.set_double_val("d1", 2.5);
  s1.set_int_val("i1", -10);
  s1.set_string_val("s1", "cow says moo");
  s1.set_complex_val("z1", complex_t(-1.5, 3.5));
}
