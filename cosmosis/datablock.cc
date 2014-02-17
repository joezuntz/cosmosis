#include "datablock.h"

using namespace std;

bool cosmosis::DataBlock::getDouble(string const& name, double& val) const noexcept
{
  auto i = values_.find(name);
  if (i == values_.end())
    return false;
  val = i->second;
  return true;
}

double cosmosis::DataBlock::get_double(string const& name) const
{
  double result;
  // The exception thrown should be more informative; this is just a
  // stub.
  return getDouble(name, result) 
    ? result 
    : throw NameNotFound(name);
}

void cosmosis::DataBlock::putDouble(string const& name, double val)
{
  if (hasValue(name)) throw NameAlreadyExists(name);
  values_[name] = val;
}

extern "C"
{
  c_datablock* make_c_datablock(void)
  {
    return new cosmosis::DataBlock();
  }

  void destroy_c_datablock(c_datablock* s)
  {
    delete static_cast<cosmosis::DataBlock*>(s);
  }

  int c_datablock_get_double(c_datablock const* s, const char* name, double* val)
  {
    auto p = static_cast<cosmosis::DataBlock const*>(s);
    bool rc = p->getDouble(name, *val);
    return rc ? 0 : 1;
  }

  int c_datablock_put_double(c_datablock* s , const char* name, double val)
  {
    auto p = static_cast<cosmosis::DataBlock*>(s);
    int rc = 0;
    try
      {
	p->putDouble(name, val);
      }
    catch (cosmosis::NameAlreadyExists const&)
      {
	rc = 1;
      }
    catch (...)
      {
	/* Log message about illegal exception here. */
	abort();
      }

    return rc;
  }
}
