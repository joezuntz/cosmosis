#include "sample.h"

using namespace std;

bool cosmosis::Sample::getDouble(string const& name, double& val) const noexcept
{
  auto i = values_.find(name);
  if (i == values_.end())
    return false;
  val = i->second;
  return true;
}

double cosmosis::Sample::get_double(string const& name) const
{
  double result;
  // The exception thrown should be more informative; this is just a
  // stub.
  return getDouble(name, result) 
    ? result 
    : throw std::runtime_error("Value lookup error");
}

void cosmosis::Sample::setDouble(string const& name, double val)
{
  values_[name] = val;
}

extern "C"
{
  c_sample* make_c_sample(void)
  {
    return new cosmosis::Sample();
  }

  void destroy_c_sample(c_sample* s)
  {
    delete static_cast<cosmosis::Sample*>(s);
  }

  int c_sample_get_double(c_sample const* s, const char* name, double* val)
  {
    auto p = static_cast<cosmosis::Sample const*>(s);
    bool rc = p->getDouble(name, *val);
    return rc ? 0 : 1;
  }

  void c_sample_set_double(c_sample* s , const char* name, double val)
  {
    auto p = static_cast<cosmosis::Sample*>(s);
    p->setDouble(name, val);
  }
}
