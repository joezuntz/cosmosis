#include "datablock.hh"

#include <cassert>

using cosmosis::DataBlock;

void test_int()
{
  DataBlock b;
  assert(not b.has_section("sect_a"));
  assert(b.put_val("sect_a", "i", 100) == DBS_SUCCESS);
  assert(b.has_section("sect_a"));
  int i;
  assert(b.get_val("sect_a", "i", i) == DBS_SUCCESS);
  assert(b.get_val("no such section", "i", i) == DBS_SECTION_NOT_FOUND);
  assert(b.get_val("sect_a", "no such parameter", i) == DBS_NAME_NOT_FOUND);

  assert(b.replace_val("sect_a", "no such parameter", i) == DBS_NAME_NOT_FOUND);
  assert(b.replace_val("no such section", "i", i) == DBS_SECTION_NOT_FOUND);
  assert(b.replace_val("sect_a", "i", 2.5) == DBS_WRONG_VALUE_TYPE);
  assert(b.replace_val("sect_a", "i", -25) == DBS_SUCCESS);
  assert(b.get_val("sect_a", "i", i) == DBS_SUCCESS);
  assert(i == -25);
}

int main()
{
  test_int();

}
