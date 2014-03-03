#ifndef COSMOSIS_DATABLOCK_HH
#define COSMOSIS_DATABLOCK_HH

//----------------------------------------------------------------------
// The class DataBlock provides the facilities to pass data from a
// Sampler to physics modules. It provides storage of and access to
// "named values". The value identification scheme is two-part: the
// full specificiation of the identifier consists of the specification
// of a 'section' and a 'name'. Values can be any of the following types:
//
//    1) int
//    2) double
//    3) std::string
//    4) std::complex<double>
//    5) std::vector<int>
//    6) std::vector<double>
//    7) std::vector<std::string>
//    8) std::vector<std::complex<double>>
//
// The class is designed to support both idiomatic C++ usage
// (recommended for C++ client code) and to support a C interface. For
// this reasons, much of the functionality of the interface is
// presented twice, once with an interface that is more convenient to
// use from C++, which is permitted to throw exceptions, and again
// with an interface that is guaranteed never to throw exceptions.
//
// The C interface does not provide direct access to the memory
// controlled by the DataBlock; it provides the user with a copy of
// the data, copied to a memory location of the user's
// specification. For any array obtained from the DataBlock, the C
// user must deallocate the array when he is done with it. The
// deallocation does not need to be done for arrays that are on the
// stack, into which DataBlock copies the data.
//
//  TODO: Add support for 2-dimensional arrays of the numeric types.
//
//----------------------------------------------------------------------

#include <string>
#include <map>

#include "datablock_status.h"
#include "section.hh"

// extern "C" {
// #include "c_datablock.h"
// }

namespace cosmosis
{
  class DataBlock
  {
  public:
    // All memory management functions are compiler generated.

    // get functions return the status, and set the value of their
    // output argument only upon success.
    template <class T>
    DATABLOCK_STATUS get_val(std::string const& section,
			     std::string const& name,
			     T& val) const;
    // put and replace functions return the status of the or
    // replace. They modify the state of the object only on success.
    template <class T>
    DATABLOCK_STATUS put_val(std::string const& section,
			     std::string const& name,
			     T const& val);

    template <class T>
    DATABLOCK_STATUS replace_val(std::string const& section,
				 std::string const& name,
				 T const& val);

    // Return true if the DataBlock has a section with the given name.
    bool has_section(std::string const& name) const;

  private:
    std::map<std::string, Section> sections_;
  };
}

template <class T>
DATABLOCK_STATUS
cosmosis::DataBlock::get_val(std::string const& section,
			     std::string const& name,
			     T& val) const
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  return isec->second.get_val(name, val);
}

template <class T>
DATABLOCK_STATUS
cosmosis::DataBlock::put_val(std::string const& section,
			     std::string const& name,
			     T const& val)
{
  auto& sec = sections_[section]; // create one if needed
  return sec.put_val(name, val);
}

template <class T>
DATABLOCK_STATUS
cosmosis::DataBlock::replace_val(std::string const& section,
				 std::string const& name,
				 T const& val)
{
  auto isec = sections_.find(section);
  if (isec == sections_.end()) return DBS_SECTION_NOT_FOUND;
  return isec->second.replace_val(name, val);
}

#endif
