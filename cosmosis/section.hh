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

    bool get(std::string const&name, double& val) const;

    template <class T>
    DATABLOCK_STATUS set_val(std::string const& name, T const& value);
    
    // DATABLOCK_STATUS set_val(std::string const& name, int val);
    // DATABLOCK_STATUS set_val(std::string const& name, double val);
    // DATABLOCK_STATUS set_val(std::string const& name, std::string const& val);
    // DATABLOCK_STATUS set_val(std::string const& name, complex_t val);
    // DATABLOCK_STATUS set_val(std::string const& name, std::vector<int> const& val);
    // DATABLOCK_STATUS set_val(std::string const& name, std::vector<double> const& val);
    // DATABLOCK_STATUS set_val(std::string const& name, std::vector<std::string> const& val);
    // DATABLOCK_STATUS set_val(std::string const& name, std::vector<complex_t> const& val);    
    
    bool has_value(std::string const& name) const;

  private:
    std::map<std::string, Entry> vals_;
  };
}

template <class T>
DATABLOCK_STATUS 
cosmosis::Section::set_val(std::string const& name, T const& v)
{
  auto i = vals_.find(name);
  if (i != vals_.end() ) return DBS_NAME_ALREADY_EXISTS;
  vals_.insert(i, make_pair(name, Entry(v)));
  return DBS_SUCCESS;
}

// template <class T>
// DATABLOCK_STATUS 
// cosmosis::Section::set_val(std::string const& name,
//                            std::initializer_list<T> val)
// {
//   return set_val(name, std::vector<T>(val));
// }

inline
bool cosmosis::Section::has_value(std::string const& name) const
{
  return (vals_.find(name) != vals_.end());
}


#endif
