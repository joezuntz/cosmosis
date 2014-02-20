#ifndef COSMOSIS_SECTION_HH
#define COSMOSIS_SECTION_HH

#include <string>
#include <map>
#include "entry.hh"

namespace cosmosis
{
  class Section
  {
  public:

    // get functions return false if the given name is not found.
    bool get(std::string const&name, double& val) const noexcept;
    

    bool set_int_val(std::string const& name, int val);
    bool set_double_val(std::string const& name, double val);
    bool set_string_val(std::string const& name, std::string val);
    bool set_complex_val(std::string const& name, complex_t val);

    bool has_value(std::string const& name) const;

  private:
    std::map<std::string, Entry> vals_;
  };
}

inline
bool cosmosis::Section::has_value(std::string const& name) const
{
  return (vals_.find(name) != vals_.end());
}


#endif
