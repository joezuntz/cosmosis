#include "datablock.hh"

using namespace std;

bool cosmosis::DataBlock::has_section(string const& name) const
{
  return sections_.find(name) != sections_.end();
}
