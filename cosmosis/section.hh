#ifndef COSMOSIS_SECTION_HH
#define COSMOSIS_SECTION_HH

#include <initializer_list>
#include <map>
#include <string>

#include "entry.hh"
#include "datablock_status.h"

namespace cosmosis
{
  // A Section represents a related set of named quantities, and
  // provides 'get', 'put', and 'replace' ability for each type of
  // quantity.
  //
  // Original author: Marc Paterno (paterno@fnal.gov)

  class Section
  {
  public:

    template <class T>
    DATABLOCK_STATUS put_val(std::string const& name, T const& value);

    template <class T>
    DATABLOCK_STATUS replace_val(std::string const& name, T const& value);

    // return true if we have a value of the right type for the given name.
    template <class T> bool has_value(std::string const& name) const;

    template <class T> 
    DATABLOCK_STATUS get_val(std::string const& name, T& v) const;

    // Return true if we have a value of any type with the given name.
    bool has_name(std::string const& name) const;

  private:
    std::map<std::string, Entry> vals_;
  };
}

template <class T>
DATABLOCK_STATUS 
cosmosis::Section::put_val(std::string const& name, T const& v)
{
  auto i = vals_.find(name);
  if (i == vals_.end() )
    {
      vals_.insert(i, make_pair(name, Entry(v)));
      return DBS_SUCCESS;
    }
  return DBS_NAME_ALREADY_EXISTS;
}

template <class T>
DATABLOCK_STATUS
cosmosis::Section::replace_val(std::string const& name, T const& v)
{
  auto i = vals_.find(name);
  if (i == vals_.end()) return DBS_NAME_NOT_FOUND;
  i->second.set_val(v);
  return DBS_SUCCESS;
}

template <class T>
bool 
cosmosis::Section::has_value(std::string const& name) const
{
  auto i = vals_.find(name);
  return (i != vals_.end()) && i->second.is<T>();
}

template <class T>
DATABLOCK_STATUS
cosmosis::Section::get_val(std::string const& name, T& v) const
{
  auto i = vals_.find(name);
  if (i == vals_.end()) return DBS_NAME_NOT_FOUND;
  if (not i->second.is<T>()) return DBS_WRONG_VALUE_TYPE;
  v = i->second.val<T>();
  return DBS_SUCCESS;  
}

#endif
