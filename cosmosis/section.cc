#include "section.hh"

using std::string;

bool
cosmosis::Section::has_val(string const& name) const
{
  return vals_.find(name) != vals_.end();
}

int
cosmosis::Section::get_size(string const& name) const
{
  auto ival = vals_.find(name);
  if (ival == vals_.end()) return -1;
  return ival->second.size();
}
