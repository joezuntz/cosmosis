#ifndef COSMOSIS_NDARRAY_HH
#define COSMOSIS_NDARRAY_HH

#include <numeric>
#include <functional>
#include <vector>

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

    std::size_t ndims() const;

    bool operator==(ndarray<T> const& other) const;

    const_iterator cbegin() { return data_.cbegin(); }
    iterator begin() { return data_.begin(); }
    const_iterator begin() const { return data_.cbegin(); }

    const_iterator cend() { return data_.cend(); }
    iterator end() { return data_.end(); }
    const_iterator end() const { return data_.cend(); }

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

#endif
