#ifndef COSMOSIS_CLAMPED_SIZE_HH
#define COSMOSIS_CLAMPED_SIZE_HH

#include <cstddef>
#include <limits>

namespace cosmosis
{
  // C++ uses the type std::size_t for 'size' related functions, and for
  // indexing into vectors and similar classes. C generally uses 'int'
  // for the same purpose.
  //
  // clamped_size is used to translate and std::size_t value into an int
  // value, being careful to deal with overflows.
  constexpr int SIZE_CONVERSION_OVERFLOW = -2;

  inline int clamp(std::size_t sz) 
  {
    return sz > std::numeric_limits<int>::max() ? SIZE_CONVERSION_OVERFLOW : sz;
  }

}


#endif
