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

#include <stdexcept>
#include <string>
#include <map>

extern "C" {
#include "c_datablock.h"
}

namespace cosmosis
{
  // Exception type thrown by DataBlock functions.
  // TODO: make the exceptions carry the corresponding enumerated error code rather than the int.
  struct Error : public std::runtime_error 
  {
    Error(std::string const& msg, DATABLOCK_STATUS errorcode ) :
      std::runtime_error(msg), code(errorcode)  
    { }
    DATABLOCK_STATUS code;
  };

  class DataBlock
  {
  public:
    // All memory management functions are compiler generated.

    // get functions return the status, and set the value of their
    // output argument only upon success.
    DATABLOCK_STATUS get(std::string const& name, double& val) const;
    DATABLOCK_STATUS get(std::string const& name, int& val) const;

    // get_X functions throw if the given name is not found.
    double get_double(std::string const& name) const;
    int    get_int(std::string const& name) const;

    // put and replace functions return the status of the or
    // replace. They modify the state of the object only on success.
    DATABLOCK_STATUS put(std::string const& name, double val);
    DATABLOCK_STATUS put(std::string const& name, int val);
    DATABLOCK_STATUS replaceDouble(std::string const& name, double val);

    // Return true if a value (of any time) with this name exists.
    bool has_value(std::string const& name) const;

  private:
    std::map<std::string, double> doubles_;
    std::map<std::string, int> ints_;
  };
}

inline
bool cosmosis::DataBlock::has_value(std::string const& name) const
{
  return (doubles_.find(name) != doubles_.end() ||
	  ints_.find(name) != ints_.end());
}

#endif
