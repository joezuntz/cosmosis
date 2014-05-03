#ifndef COSMOSIS_SECTION_HH
#define COSMOSIS_SECTION_HH

#include <initializer_list>
#include <map>
#include <string>

#include "exceptions.hh"
#include "entry.hh"
#include "datablock_status.h"
#include "datablock_types.h"

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
    struct BadSectionAccess : cosmosis::Exception { }; // used for exceptions

    template <class T>
    DATABLOCK_STATUS put_val(std::string const& name, T const& value);

    template <class T>
    DATABLOCK_STATUS replace_val(std::string const& name, T const& value);

    // return true if we have a value of the right type for the given name.
    template <class T> bool has_value(std::string const& name) const;

    //Return the number of items stored in this section
    std::size_t number_values() const;

    // Return -1 if this section has no parameter with the given name,
    // or if the parameter is not an array type. Return -2 if the array
    // length is longer than MAXINT. Otherwise, return the length of the
    // array.
    int get_size(std::string const& name) const;
    
    DATABLOCK_STATUS 
    get_type(std::string const&name, datablock_type_t &t) const;

    // Return DBS_SUCCESS if the value with the given name is of the
    // given type, and and error if there is no such name or the value
    // is not of the given type. If returning DBS_SUCCESS, the value of
    // v is set; otherwise the value is not defined.
    template <class T>
    DATABLOCK_STATUS get_val(std::string const& name, T& v) const;

    // Return DBS_SUCCESS if the value with the given name is of the
    // given type, or if there is no value with the given name. Return
    // an error if the value associated with the given name is of the
    // wrong type. If the name was not found, set v to be the supplied
    // default; otherwise, set v to be the value associated with the
    // name.
    template <class T>
    DATABLOCK_STATUS get_val(std::string const& name, T const& def, T& v) const;

    // Return true if we have a value of any type with the given name.
    bool has_val(std::string const& name) const;

    // Return DBS_SUCCESS if the value with the given name is of the
    // given type, and is an array, and an error if there is no such
    // name, or if the named value is not an array of the given type. If
    // returning DBS_SUCESS, set 'extents' to carry the extent of each
    // dimension of the array.
    template <class T>
    DATABLOCK_STATUS get_array_shape(std::string const& name,
                                     std::vector<std::size_t>& extents) const;

    //Return the name of the key at position i
    std::string const& value_name(std::size_t i) const;

    // The view functions provide readonly access to the data in the
    // Section without copying the data. The reference returned by a
    // call to view is invalidated if any replace function is called for
    // the same name. Throws BadSectionAccess if the name can't be
    // found, and BadEntry if the contained value is the wrong type.
    template <class T>
    T const& view(std::string const& name) const;

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
      vals_.emplace(name, Entry(v));
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
  if (not i->second.is<T>()) return DBS_WRONG_VALUE_TYPE;
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

template <class T>
DATABLOCK_STATUS
cosmosis::Section::get_val(std::string const& name, T const& def, T& v) const
{
  auto i = vals_.find(name);
  if (i == vals_.end())
    {
      v = def;
      return DBS_USED_DEFAULT;
    }
  if (not i->second.is<T>()) return DBS_WRONG_VALUE_TYPE;
  v = i->second.val<T>();
  return DBS_SUCCESS;
}

template <class T>
DATABLOCK_STATUS
cosmosis::Section::get_array_shape(std::string const& name,
                                   std::vector<std::size_t>& extents) const
{
  auto i = vals_.find(name);
  if (i == vals_.end()) return DBS_NAME_NOT_FOUND;
  typedef ndarray<T> array_t;
  if (not i->second.is<array_t>()) return DBS_WRONG_VALUE_TYPE;
  auto const& r = view<array_t>(name);
  extents = r.extents();
  return DBS_SUCCESS;
}

template <class T>
T const&
cosmosis::Section::view(std::string const& name) const
{
  auto i = vals_.find(name);
  if (i == vals_.end()) throw BadSectionAccess();
  return i->second.view<T>();
}

#endif
