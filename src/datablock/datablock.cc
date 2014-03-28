#include "datablock.hh"

using namespace std;

bool cosmosis::DataBlock::has_val(string const& section,
                                              string const& name) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return false;
  return isec->second.has_val(name) ? true : false;
}

int cosmosis::DataBlock::get_size(string const& section,
                                  string const& name) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return -1;
  return isec->second.get_size(name);
}

DATABLOCK_STATUS cosmosis::DataBlock::get_type(string const& section,
                                              string const& name, datablock_type_t &t) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  return isec->second.get_type(name,t);
}


bool cosmosis::DataBlock::has_section(string const& name) const
{
  return sections_.find(name) != sections_.end();
}

DATABLOCK_STATUS cosmosis::DataBlock::get_number_values(string const& section, int &n) const
{
  n = -1;
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  n = isec->second.number_values();
  return DBS_SUCCESS;
}


std::size_t cosmosis::DataBlock::num_sections() const
{
  return sections_.size();
}

std::string const& cosmosis::DataBlock::section_name(std::size_t i) const
{
  if (i >= num_sections()) throw BadDataBlockAccess();
  auto isec = sections_.begin();
  std::advance(isec, i);
  return isec->first;

}

void cosmosis::DataBlock::clear()
{
  sections_.clear();
}
