#ifndef COSMOSIS_NDARRAY_HH
#define COSMOSIS_NDARRAY_HH

#include <exception>
#include <functional>
#include <numeric>
#include <vector>

#include "exceptions.hh"

namespace cosmosis {

  // NDArrayIndexException is thrown when an ndarray<T> object is indexed with
  // the wrong number of indices, e.g. a 2-D array is accessed using 1 or 3
  // indices.
  class NDArrayIndexException : public cosmosis::Exception {
  public:
    virtual const char*
    what() const throw()
    {
      return "Wrong number of indices to NDArray";
    }
  };

  // Calculate the number of elements in an array with the given set of extents.
  inline std::size_t
  num_elements(std::vector<std::size_t> const& extents)
  {
    return std::accumulate(
      extents.begin(), extents.end(), 1, std::multiplies<std::size_t>());
  }

  // ndarray<T> represents an n-dimensional array of type T. Note the type does
  // not determine the dimensionality of the array, nor the extent of the array
  // in each dimension; these are run-time values that are set on construction
  // and can not be modified thereafter. The use of template parameters to fix
  // the dimensionality at compile-time is avoided, so that it is easier to use
  // ndarray<T> from C, Fortran, and Python.
  template <typename T>
  class ndarray {
  public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    // Create an ndarray with the given number of dimensions, and the
    // given extent for each dimension. 'extents' must be the start of
    // an array of extent at least as great as ndims; only the first
    // ndims elements will be read.
    ndarray(T const* data, int ndims, int const* extents);

    // Create an ndarray carrying the given data, with the given
    // extents. This construction is intended primarily for testing.
    // The arguments are passed by value to take advantage of move
    // construction, when possible.
    ndarray(std::vector<T> vals, std::vector<std::size_t> extents);

    std::size_t ndims() const;

    bool operator==(ndarray<T> const& other) const;

    const_iterator cbegin() const;
    iterator begin();
    const_iterator begin() const;

    const_iterator cend() const;
    iterator end();
    const_iterator end() const;

    std::vector<std::size_t> const& extents() const;

    // size() returns the number of elements in the array, which is the product
    // of all the extents.
    std::size_t size() const;

    // general n-D element access - LHS
    // Use of this function on an ndarray "x" looks like:
    //
    //     x(i,j,k) = val ; // this assumes x is an ndarray with dimension 3.
    //
    template <typename... Args>
    T& operator()(Args... indices);

    // general n-D elemetn access - RHS
    // Use of this function on an ndarray "x" looks like:
    //
    //   auto val = x(i, j, k); // this assumes x is an ndarray with dimension 3
    template <typename... Args>
    T operator()(Args... indices) const;

  private:
    std::vector<std::size_t> extents_;
    std::vector<T> data_;

    template <typename... Args>
    size_t get_index(Args... indices);
  };
}

template <typename T>
cosmosis::ndarray<T>::ndarray(T const* data, int ndims, int const* extents)
  : extents_(extents, extents + ndims)
  , data_(data, data + num_elements(extents_))
{}

template <typename T>
cosmosis::ndarray<T>::ndarray(std::vector<T> vals,
                              std::vector<std::size_t> extents)
  : extents_(std::move(extents)), data_(std::move(vals))
{}

template <typename T>
std::size_t
cosmosis::ndarray<T>::ndims() const
{
  return extents_.size();
}

template <typename T>
bool
cosmosis::ndarray<T>::operator==(ndarray<T> const& other) const
{
  return (extents_ == other.extents_ && data_ == other.data_);
}

template <typename T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::cbegin() const
{
  return data_.cbegin();
}

template <typename T>
typename cosmosis::ndarray<T>::iterator
cosmosis::ndarray<T>::begin()
{
  return data_.begin();
}

template <typename T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::begin() const
{
  return data_.begin();
}

template <typename T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::cend() const
{
  return data_.cend();
}

template <typename T>
typename cosmosis::ndarray<T>::iterator
cosmosis::ndarray<T>::end()
{
  return data_.end();
}

template <typename T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::end() const
{
  return data_.end();
}

template <typename T>
std::vector<std::size_t> const&
cosmosis::ndarray<T>::extents() const
{
  return extents_;
}

template <typename T>
std::size_t
cosmosis::ndarray<T>::size() const
{
  return data_.size();
}

template <typename T>
template <typename... Args>
T&
cosmosis::ndarray<T>::operator()(Args... indices)
{
  return data_[get_index(indices...)];
}

template <typename T>
template <typename... Args>
T
cosmosis::ndarray<T>::operator()(Args... indices) const
{
  return data_[get_index(indices...)];
}

// Private member functions.

// TODO: Look at generated assembly code to see if replacing the for loop with
// compile-time recursion makes any difference.
template <typename T>
template <typename... Args>
std::size_t
cosmosis::ndarray<T>::get_index(Args... indices)
{
  constexpr size_t NDIMS = sizeof...(indices);
  if (NDIMS != ndims())
    throw NDArrayIndexException();

  const size_t indices2[NDIMS] = {indices...};
  size_t index1D = 0;
  size_t r = 1;
  for (size_t i = 0; i < NDIMS; i++) {
    index1D += indices2[NDIMS - 1 - i] * r;
    r *= extents_[NDIMS - 1 - i];
  }
  return index1D;
}

#endif
