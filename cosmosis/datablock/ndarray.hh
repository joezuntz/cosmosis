#ifndef COSMOSIS_NDARRAY_HH
#define COSMOSIS_NDARRAY_HH

#include <numeric>
#include <functional>
#include <vector>
#include <exception>

class NDArrayIndexException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Wrong number of indices to NDArray";
  }
} ndarrayindexexception;


inline
std::size_t num_elements(std::vector<std::size_t> const& extents)
{
  return std::accumulate(extents.begin(), extents.end(), 1,
                         std::multiplies<std::size_t>());
}

namespace cosmosis
{
  template <class T> class ndarray
  {
  public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;

    // Create an ndarray with the given number of dimensions, and the
    // given extent for each dimension. 'extents' must be the start of
    // an array of extent at least as great as ndims; only the first
    // ndims elements will be read.
    ndarray(T const* data, int const& ndims, int const* extents);

    // Create an ndarray carrying the given data, with the given
    // extents. This construction is intended primarily for testing.
    // The arguments are passed by value to take advantage of move
    // construction, when possible.
    ndarray(std::vector<T> vals, std::vector<std::size_t> extents);

    std::size_t ndims() const;

    bool operator==(ndarray<T> const& other) const;

    const_iterator cbegin();
    iterator begin();
    const_iterator begin() const;

    const_iterator cend();
    iterator end();
    const_iterator end() const;

    std::vector<std::size_t> const& extents() const;

    // size() returns the number of elements in the array.
    std::size_t size() const;

    // Access to data elements.
    // No bounds checking!
    // We do two specialist cases for 2D and 3D as they will 
    // be the most common, and this might be slightly faster

    // 2D element access - version for RHS a = x(i,j)
    const T& operator()(int i, int j) const{
      return data_[extents_[1]*i+j];
    }

    // 2D element access - version for LHS x(i,j) = 1
    T& operator()(int i, int j){
      return data_[extents_[1]*i+j];
    }


    // 3D element access RHS
    const T& operator()(int i, int j, int k) const{
      return data_[extents_[2]*extents_[1]*i+extents_[1]*j+k];
    }

    // 3d element access LHS
    T& operator()(int i, int j, int k){
      return data_[extents_[2]*extents_[1]*i+extents_[1]*j+k];
    }

    // Indexing for general nd array
    // the "..." thing is called a "parameter pack"
    // and is a C++11 thing
    template<typename... Args>
    size_t get_index(Args... indices){
      const size_t n = sizeof...(indices);
      size_t indices2[n] = {indices...};
      if (n!=ndims()) throw ndarrayindexexception;
      size_t index1D = 0;
      size_t r = 1;
      for (size_t i=0; i<n; i++){
        index1D += indices2[n-1-i] * r;
        r *= extents_[n-1-i];
      }
      return index1D;      
    }

    // general n-D element access - LHS
    template<typename... Args>
    T& operator()(Args... indices)
    {
      return data_[get_index(indices...)];
    }

    // general n-D elemetn access - RHS
    template<typename... Args>
    const T& operator()(Args... indices) const
    {
      return data_[get_index(indices...)];
    }



  private:
    std::vector<std::size_t> extents_;
    std::vector<T> data_;
  };
}

template <class T>
cosmosis::ndarray<T>::ndarray(T const* data, int const& ndims, int const* extents) :
  extents_(extents, extents+ndims),
  data_(data, data + num_elements(extents_))
{
}

template <class T>
cosmosis::ndarray<T>::ndarray(std::vector<T> vals, std::vector<std::size_t> extents) :
  extents_(std::move(extents)),
  data_(std::move(vals))
{ }

template <class T>
std::size_t
cosmosis::ndarray<T>::ndims() const
{
  return extents_.size();
}

template <class T>
bool
cosmosis::ndarray<T>::operator==(ndarray<T> const& other) const
{
  return (extents_ == other.extents_ && data_ == other.data_);
}

template <class T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::cbegin()
{
  return data_.cbegin();
}

template <class T>
typename cosmosis::ndarray<T>::iterator
cosmosis::ndarray<T>::begin()
{
  return data_.begin();
}

template <class T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::begin() const
{
  return data_.begin();
}

template <class T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::cend()
{
  return data_.cend();
}

template <class T>
typename cosmosis::ndarray<T>::iterator
cosmosis::ndarray<T>::end()
{
  return data_.end();
}

template <class T>
typename cosmosis::ndarray<T>::const_iterator
cosmosis::ndarray<T>::end() const
{
  return data_.end();
}

template <class T>
std::vector<std::size_t> const&
cosmosis::ndarray<T>::extents() const
{
  return extents_;
}

template <class T>
std::size_t
cosmosis::ndarray<T>::size() const
{
  return data_.size();
}

#endif
