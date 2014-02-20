#include "datablock.hh"

using namespace std;

DATABLOCK_STATUS
cosmosis::DataBlock::get(string const& name, double& val) const
{
  auto i = doubles_.find(name);
  if (i == doubles_.end()) return DBS_NAME_NOT_FOUND;
  val = i->second;
  return DBS_SUCCESS;
}

DATABLOCK_STATUS
cosmosis::DataBlock::get(string const& name, int& val) const
{
  auto i = ints_.find(name);
  if (i == ints_.end()) return DBS_NAME_NOT_FOUND;
  val = i->second;
  return DBS_SUCCESS;
}


double cosmosis::DataBlock::get_double(string const& name) const
{
  double result;
  DATABLOCK_STATUS rc = get(name, result);
  if (rc != DBS_SUCCESS) throw Error(name, rc);
  return result;
}

DATABLOCK_STATUS cosmosis::DataBlock::put(string const& name, double val)
{
  if (has_value(name)) return DBS_NAME_ALREADY_EXISTS;
  doubles_[name] = val;
  return DBS_SUCCESS;
}

DATABLOCK_STATUS cosmosis::DataBlock::put(string const& name, int val)
{
  if (has_value(name)) return DBS_NAME_ALREADY_EXISTS;
  ints_[name] = val;
  return DBS_SUCCESS;
}
