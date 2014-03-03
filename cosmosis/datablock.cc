#include "datablock.hh"

using namespace std;

bool cosmosis::DataBlock::has_section(string const& name) const
{
  return sections_.find(name) != sections_.end();
}

std::size_t cosmosis::DataBlock::num_sections() const
{
  return sections_.size();
}

void cosmosis::DataBlock::clear()
{
  sections_.clear();
}
