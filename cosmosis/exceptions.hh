#ifndef COSMOSIS_EXCEPTIONS_HH
#define COSMOSIS_EXCEPTIONS_HH

namespace cosmosis
{
  // This is the base class for all exceptions thrown by cosmosis
  // functions.
  struct Exception { virtual ~Exception() = default; } ;
}

#endif
