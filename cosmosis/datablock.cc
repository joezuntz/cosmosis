#include "datablock.hh"

using namespace std;

DATABLOCK_STATUS cosmosis::DataBlock::has_val(string const& section,
                                              string const& name) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  return isec->second.has_val(name) ? DBS_SUCCESS : DBS_NAME_NOT_FOUND;
}

bool cosmosis::DataBlock::has_section(string const& name) const
{
  return sections_.find(name) != sections_.end();
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
