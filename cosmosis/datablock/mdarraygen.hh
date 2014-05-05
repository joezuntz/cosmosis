#ifndef COSMOSIS_MD_ARRAY_GEN_HH
#define COSMOSIS_MD_ARRAY_GEN_HH

#include <cstddef>
#include <vector>

// MDArrayGen is a compile-time function used to generate several types.
// These types are all expressed as nested types within MDArrayGen<...>
//   'type' is a typedef for a native n-dimensional array.
//   'reference' is a typedef for a reference to 'type'
//   'pointer' is a typedef for a pointer to 'type'
//
// MDArrayGen also contains several compile-time data values:
//   'ndims' is the number of dimensions of the array 'type'
//   'extent' is the length of the leading dimension of 'type'
//   'size_bytes' is the size of the array, in bytes
//   'size_elements' is the number of elements in the array
// 
// Finally, MDArrayGen provides a static member function used to
// determine whether its associated 'type' matches a given vector of
// extents, as would be supplied by cosmosis::ndarray<T>.
//
template <typename T, std::size_t ... args> struct MDArrayGen;

template <typename T> struct MDArrayGen<T>
{
  typedef T type;

  static constexpr std::size_t ndims = 0;
  static constexpr std::size_t extent = 0;
  static constexpr std::size_t size_bytes = sizeof(T);
  static constexpr std::size_t size_elements = 1;

  //template< typename Container > static bool matches(Container const&) { return true; }

  static bool matches(std::vector<std::size_t> const&) { return true; }

};

template <typename T, std::size_t first, std::size_t ... args> struct MDArrayGen<T,first,args...>
{
  using inner_t = MDArrayGen<T, args...>;
  //typedef MDArrayGen<T, args...> inner_t;

  using type = typename inner_t::type[first];
  //  typedef typename inner_t::type type[first];

  using pointer = typename inner_t::type(*)[first];
  //using pointer = type*;
  //typedef type* pointer;
  //typedef typename inner_t::type (*pointer)[first];

  using reference = typename inner_t::type(&)[first];
  //using reference = type&;
  //typedef type& reference;
  //typedef typename inner_t::type (&reference)[first];

  static constexpr std::size_t ndims = sizeof...(args) + 1;
  static constexpr std::size_t extent = first;
  static constexpr std::size_t size_bytes = first * MDArrayGen<T, args...>::size_bytes;
  static constexpr std::size_t size_elements = first * MDArrayGen<T, args...>::size_elements;

  static bool matches(std::vector<std::size_t> const& extents)
  {
    //constexpr std::size_t index = ndims+1;
    constexpr std::size_t index = ndims-1;
    //constexpr std::size_t index = inner_t::ndims;
    return first == extents[index] && inner_t::matches(extents);
  }

  // template <typename Container>
  // static bool matches(Container const& extents)
  // {
  //   const std::size_t index = ndims-1;
  //   return first == extents[index] && inner_t::matches(extents);
  // }
};

#endif
