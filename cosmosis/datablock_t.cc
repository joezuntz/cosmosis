#include "datablock.hh"

#include <cassert>

using cosmosis::DataBlock;

int main()
{
  DataBlock b;
  assert(not b.has_section("sect_a"));
}
