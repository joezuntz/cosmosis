#include "datablock.hh"

using namespace std;

bool cosmosis::DataBlock::has_val(string section,
                                              string name) const
{
  downcase(section); downcase(name);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return false;
  return isec->second.has_val(name) ? true : false;
}

int cosmosis::DataBlock::get_size(string section,
                                  string name) const
{
  downcase(section); downcase(name);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return -1;
  return isec->second.get_size(name);
}

DATABLOCK_STATUS cosmosis::DataBlock::get_type(string section,
                                              string name, datablock_type_t &t) const
{
  downcase(section); downcase(name);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  return isec->second.get_type(name,t);
}


bool cosmosis::DataBlock::has_section(string name) const
{
  downcase(name);
  return sections_.find(name) != sections_.end();
}

int cosmosis::DataBlock::num_values(string const& section) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return -1;
  return isec->second.number_values();
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


std::string const& cosmosis::DataBlock::value_name(int i, int j) const
{
  std::string section = section_name(i);
  return value_name(section,j);
}


std::string const& cosmosis::DataBlock::value_name(std::string section, int j) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) throw BadDataBlockAccess();
  return isec->second.value_name(j);
}


void cosmosis::DataBlock::clear()
{
  sections_.clear();
}

DATABLOCK_STATUS 
cosmosis::DataBlock::delete_section(std::string section)
{
  downcase(section);
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  sections_.erase(isec);
  return DBS_SUCCESS;
}

