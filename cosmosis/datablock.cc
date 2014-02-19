#include "datablock.h"

using namespace std;

bool cosmosis::DataBlock::get(string const& name, double& val) const noexcept
{
  auto i = doubles_.find(name);
  if (i == doubles_.end())
    return false;
  val = i->second;
  return true;
}

bool cosmosis::DataBlock::get(string const& name, int& val) const noexcept
{
  auto i = ints_.find(name);
  if (i == ints_.end())
    return false;
  val = i->second;
  return true;
}


double cosmosis::DataBlock::get_double(string const& name) const
{
  double result;
  // The exception thrown should be more informative; this is just a
  // stub.
  return get(name, result)
    ? result
    : throw NameNotFound(name);
}

void cosmosis::DataBlock::put(string const& name, double val)
{
  if (has_value(name)) throw NameAlreadyExists(name);
  doubles_[name] = val;
}

void cosmosis::DataBlock::put(string const& name, int val)
{
  if (has_value(name)) throw NameAlreadyExists(name);
  ints_[name] = val;
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
    bool rc = p->get(name, *val);
    return rc ? 0 : 1;
  }

  int c_datablock_get_int(c_datablock const* s, const char* name, int* val)
  {
    auto p = static_cast<cosmosis::DataBlock const*>(s);
    bool rc = p->get(name, *val);
    return rc ? 0 : 1;
  }

  int c_datablock_put_double(c_datablock* s , const char* name, double val)
  {
    auto p = static_cast<cosmosis::DataBlock*>(s);
    int rc = 0;
    try { p->put(name, val); }
    catch (cosmosis::NameAlreadyExists const&) { rc = 1; }
    catch (...) { abort(); }
    return rc;
  }

  int c_datablock_put_int(c_datablock* s , const char* name, int val)
  {
    auto p = static_cast<cosmosis::DataBlock*>(s);
    int rc = 0;
    try { p->put(name, val); }
    catch (cosmosis::NameAlreadyExists const&) { rc = 1; }
    catch (...) { abort(); }
    return rc;
  }

}
