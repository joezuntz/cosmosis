#include "section.hh"

using std::string;

bool
cosmosis::Section::has_name(string const& name) const
{
  return vals_.find(name) != vals_.end();
}
