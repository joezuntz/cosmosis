#ifndef COSMOSIS_FAKE_ARRAY_HH
#define COSMOSIS_FAKE_ARRAY_HH

#include <cstddef>
#include <stdexcept>

#include "mdarraygen.hh"

// FakeArray is a class template that can be used as if it were a native
// multidimensional array. However, the data manipulated by a FakeArray
// instance are stored in a one-dimensional array, held elsewhere, and
// not managed by the FakeArray instance.

template <typename T, std::size_t ... args> struct FakeArray
{
  using generator = MDArrayGen<T,args...>;
  static constexpr std::size_t size_bytes = generator::size_bytes;

  // Construct a FakeArray the refers to the given data. Do not copy the
  // data. It is illegal to call this with a null pointer.
  FakeArray(T* data);

  // This operator provides the function that allows the FakeArray to be
  // used 'as if' it were a native array.
  operator typename generator::reference();

  // Our data member carries the reference to the generated array type.
  typename generator::reference data;
};

template <typename T, std::size_t ... args>
FakeArray<T, args...>::FakeArray(T* d) :
  data(reinterpret_cast<typename generator::reference>(*d))
{ }

template <typename T, std::size_t ... args>
FakeArray<T, args...>::operator typename generator::reference()
{
  return data;
}

//
template <typename T, std::size_t ... args>
auto make_fake_array(void* data) -> typename MDArrayGen<T,args...>::reference
{
  typedef MDArrayGen<T,args...> generator;
  return (typename generator::reference)(*static_cast<char*>(data));
}

template <std::size_t ... args, typename Container>
bool verifyExtents(Container const& extents)
{
  typedef MDArrayGen<char,args...> generator;
  if (generator::ndims != extents.size()) return false;
  Container rev(extents);
  reverse(rev.begin(), rev.end());
  return generator::matches(rev);
}

#endif
