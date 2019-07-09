#ifndef COSMOSIS_ENTRY_HH
#define COSMOSIS_ENTRY_HH

#include <string>
#include <complex>
#include <vector>

#include "ndarray.hh"
#include "exceptions.hh"
#include "datablock_types.h"

// Entry is a discriminated union, capable of holding any one of the
// value types indicated by tag_t.
//
// This is an extended version of the example from section 8.3.2 in
// "The C++ Programming Language, 4th edition" by Bjarne Stroustrup.
//
// Original author: Marc Paterno (paterno@fnal.gov)
//
// Design ideas not taken:
//
// Functions that do not throw are not all declared 'noexcept', because
// there seems to be no clear advantage to doing so.
//
// strings and vectors are returned by value, rather than by const
// reference, for two reasons:
//
//    1. The requirements of the C interface to be supported include a
// requirement of copying data into user-supplied string or array
// buffers.
//
//    2. Returning a reference to the internal state of an Entry object,
//    when that internal state can change, seems error-prone.
//
// It does not seem to make sense to have a default parameter type, so
// Entry has no default constructor.
//
//  The 'is<type>' functions are present mostly so that the C interface
//  can be written in a way that can be certain to never allow an
//  exception to be thrown, without needing pervasive try/catch blocks.
//
// TODO:
//
//   1. Evaluate whether move c'tor and move assignment should be
//   supported.
//   2. Extend to support 2-dimensional arrays.
//

namespace cosmosis
{
  typedef std::complex<double> complex_t;
  typedef std::vector<int> vint_t;
  typedef std::vector<double> vdouble_t;
  typedef std::vector<std::string> vstring_t;
  typedef std::vector<complex_t> vcomplex_t;
  typedef cosmosis::ndarray<int> nd_int_t;
  typedef cosmosis::ndarray<double> nd_double_t;
  typedef cosmosis::ndarray<complex_t> nd_complex_t;

  class Entry
  {
  public:
    struct BadEntry : public cosmosis::Exception { }; // used for exceptions.

    // A default-constructed Entry carries a double, with value 0.0
    Entry();
    explicit Entry(int v);
    explicit Entry(bool v);
    explicit Entry(double v);
    explicit Entry(const char * v);
    explicit Entry(std::string v);
    explicit Entry(complex_t v);
    explicit Entry(vint_t const& a);
    explicit Entry(vdouble_t const& a);
    explicit Entry(vstring_t const& a);
    explicit Entry(vcomplex_t const& a);
    explicit Entry(nd_int_t const& a);
    explicit Entry(nd_double_t const& a);
    explicit Entry(nd_complex_t const& a);

    Entry(Entry const& other);
    Entry& operator=(Entry const& other);

    ~Entry();

    // Two Entries are equal if they carry the same type, and the same value.
    bool operator==(Entry const& other) const;

    // Return true if the Entry is currently carrying a value of type T.
    template <class T> bool is() const;

    // If the Entry is carrying a value of type T, return a copy of it.
    // Otherwise throw a BadEntry exception.
    template <class T> T val() const;

    // If the Entry is carrying a value of type T, return a reference to
    // it. Otherwise throw a BadEntry exception.
    template <class T> T const& view() const;

    // If the Entry is carrying a value that is a vector, return the
    // length of the vector. Otherwise, return -1. If the length of the
    // vector is greater than MAXINT, return -2.
    int size() const;

    // Replace the existing value (of whatever type) with the given
    // value.
    void set_val(bool v);
    void set_val(int v);
    void set_val(double v);
    void set_val(std::string const& v);
    void set_val(const char * v);
    void set_val(complex_t v);
    void set_val(vint_t const& v);
    void set_val(vdouble_t const& v);
    void set_val(vstring_t const& v);
    void set_val(vcomplex_t const& v);
    void set_val(nd_int_t const& v);
    void set_val(nd_double_t const& v);
    void set_val(nd_complex_t const& v);

  private:
    // The type of the value currenty active.
    datablock_type_t type_;

    // The anonymous union contains the value. We have a named union
    // member for each type we can hold.
    union
    {
      // scalars
      bool b;
      int i;
      double d;
      std::string s;
      complex_t z;
      // 1-d arrays
      vint_t vi;
      vdouble_t vd;
      vstring_t vs;
      vcomplex_t vz;
      // multi-dimensional arrays
      nd_int_t   ndi;
      nd_double_t ndd;
      nd_complex_t ndz;
    }; // union

    // Call the destructor of the current value, if it is a managed type.
    void _destroy_if_managed();

    // If the Entry is carrying a value of type T, return a reference to
    // it; otherwise throw BadEntry.
    template <class T> T const& _val(T* v) const;

    template <class T> datablock_type_t enum_for_type() const;

    // Set the carried value to be of type T, with value val. Use this
    // function to set types with trivial destructors.
    template <class T> void _set(T val, T& member);

    // Set the carried value to be of type T, with value val. Use this
    // function to set types with nontrival destructors.
    template <class T> void _vset(T const& val, T& member);
  }; // class Entry

  // emplace is used to do placement new of type T, with value val, at
  // location addr.
  template <class T> void emplace(T* addr, T const& val);
} // namespace cosmosis


// Implementation of member functions.
inline
cosmosis::Entry::Entry() :
  type_(DBT_DOUBLE), d(0.0)
{ }

inline
cosmosis::Entry::Entry(int v) :
  type_(DBT_INT), i(v)
{}

inline
cosmosis::Entry::Entry(bool v) :
  type_(DBT_BOOL), b(v)
{}

inline
cosmosis::Entry::Entry(double v) :
  type_(DBT_DOUBLE), d(v)
{}

inline
cosmosis::Entry::Entry(const char * v) :
  type_(DBT_STRING), s(v)
{}

inline
cosmosis::Entry::Entry(std::string v) :
  type_(DBT_STRING), s(v)
{}

inline
cosmosis::Entry::Entry(complex_t v) :
  type_(DBT_COMPLEX), z(v)
{}

inline
cosmosis::Entry::Entry(vint_t const& v) :
  type_(DBT_INT1D), vi(v)
{}

inline
cosmosis::Entry::Entry(vdouble_t const& v) :
  type_(DBT_DOUBLE1D), vd(v)
{}

inline
cosmosis::Entry::Entry(vstring_t const& v) :
  type_(DBT_STRING1D), vs(v)
{}

inline
cosmosis::Entry::Entry(vcomplex_t const& v) :
  type_(DBT_COMPLEX1D), vz(v)
{}

inline
cosmosis::Entry::Entry(nd_int_t const& v) :
  type_(DBT_INTND), ndi(v)
{}

inline
cosmosis::Entry::Entry(nd_double_t const& v) :
  type_(DBT_DOUBLEND), ndd(v)
{}

inline
cosmosis::Entry::Entry(nd_complex_t const& v) :
  type_(DBT_COMPLEXND), ndz(v)
{}

template <class T>
T const& cosmosis::Entry::_val(T* v) const
{
  if (type_ != enum_for_type<T>()) throw BadEntry();
  return *v;
}

template <class T>
void cosmosis::Entry::_set(T val, T& member)
{
  _destroy_if_managed();
  type_ = enum_for_type<T>();
  member = val;
}

template <class T>
void cosmosis::Entry::_vset(T const& val, T& member)
{
  if (type_ == enum_for_type<T>())
    member = val;
  else
    {
      _destroy_if_managed();
      type_ = enum_for_type<T>();
      emplace(&member, val);
    }
}


namespace cosmosis
{


  template <> inline datablock_type_t Entry::enum_for_type<bool const>()         const {return DBT_BOOL;}
  template <> inline datablock_type_t Entry::enum_for_type<int const>()          const {return DBT_INT;}
  template <> inline datablock_type_t Entry::enum_for_type<double const>()       const {return DBT_DOUBLE;}
  template <> inline datablock_type_t Entry::enum_for_type<std::string const>()  const {return DBT_STRING;}
  template <> inline datablock_type_t Entry::enum_for_type<complex_t const>()    const {return DBT_COMPLEX;}
  template <> inline datablock_type_t Entry::enum_for_type<vint_t const>()       const {return DBT_INT1D;}
  template <> inline datablock_type_t Entry::enum_for_type<vdouble_t const>()    const {return DBT_DOUBLE1D;}
  template <> inline datablock_type_t Entry::enum_for_type<vstring_t const>()    const {return DBT_STRING1D;}
  template <> inline datablock_type_t Entry::enum_for_type<vcomplex_t const>()   const {return DBT_COMPLEX1D;}
  template <> inline datablock_type_t Entry::enum_for_type<nd_int_t const>()     const {return DBT_INTND;}
  template <> inline datablock_type_t Entry::enum_for_type<nd_double_t const>()  const {return DBT_DOUBLEND;}
  template <> inline datablock_type_t Entry::enum_for_type<nd_complex_t const>() const {return DBT_COMPLEXND;}

  template <> inline datablock_type_t Entry::enum_for_type<bool>()         const {return DBT_BOOL;}
  template <> inline datablock_type_t Entry::enum_for_type<int>()          const {return DBT_INT;}
  template <> inline datablock_type_t Entry::enum_for_type<double>()       const {return DBT_DOUBLE;}
  template <> inline datablock_type_t Entry::enum_for_type<std::string>()  const {return DBT_STRING;}
  template <> inline datablock_type_t Entry::enum_for_type<complex_t>()    const {return DBT_COMPLEX;}
  template <> inline datablock_type_t Entry::enum_for_type<vint_t>()       const {return DBT_INT1D;}
  template <> inline datablock_type_t Entry::enum_for_type<vdouble_t>()    const {return DBT_DOUBLE1D;}
  template <> inline datablock_type_t Entry::enum_for_type<vstring_t>()    const {return DBT_STRING1D;}
  template <> inline datablock_type_t Entry::enum_for_type<vcomplex_t>()   const {return DBT_COMPLEX1D;}
  template <> inline datablock_type_t Entry::enum_for_type<nd_int_t>()     const {return DBT_INTND;}
  template <> inline datablock_type_t Entry::enum_for_type<nd_double_t>()  const {return DBT_DOUBLEND;}
  template <> inline datablock_type_t Entry::enum_for_type<nd_complex_t>() const {return DBT_COMPLEXND;}
  // template <> inline bool Entry::val<bool>() const { return _val(&b); }



  template <class T> void emplace(T* addr, T const& val) { new(addr) T(val); }
  template <class T> bool Entry::is() const { return (type_ == enum_for_type<T>()); }

  template <> inline bool Entry::val<bool>() const { return _val(&b); }
  template <> inline int Entry::val<int>() const { return _val(&i); }
  template <> inline double Entry::val<double>() const { return _val(&d); }
  template <> inline std::string Entry::val<std::string>() const { return _val(&s); }
  template <> inline complex_t Entry::val<complex_t>() const { return _val(&z); }
  template <> inline vint_t Entry::val<vint_t>() const { return _val(&vi); }
  template <> inline vdouble_t Entry::val<vdouble_t>() const { return _val(&vd); }
  template <> inline vstring_t Entry::val<vstring_t>() const { return _val(&vs); }
  template <> inline vcomplex_t Entry::val<vcomplex_t>() const { return _val(&vz); }
  template <> inline nd_int_t Entry::val<nd_int_t>() const { return _val(&ndi); }
  template <> inline nd_double_t Entry::val<nd_double_t>() const { return _val(&ndd); }
  template <> inline nd_complex_t Entry::val<nd_complex_t>() const { return _val(&ndz); }

  template <> inline bool const& Entry::view<bool>() const { return _val(&b); }
  template <> inline int const& Entry::view<int>() const { return _val(&i); }
  template <> inline double const& Entry::view<double>() const { return _val(&d); }
  template <> inline std::string const& Entry::view<std::string>() const { return _val(&s); }
  template <> inline complex_t const& Entry::view<complex_t>() const { return _val(&z); }
  template <> inline vint_t const& Entry::view<vint_t>() const { return _val(&vi); }
  template <> inline vdouble_t const& Entry::view<vdouble_t>() const { return _val(&vd); }
  template <> inline vstring_t const& Entry::view<vstring_t>() const { return _val(&vs); }
  template <> inline vcomplex_t const& Entry::view<vcomplex_t>() const { return _val(&vz); }
  template <> inline nd_int_t const& Entry::view<nd_int_t>() const { return _val(&ndi); }
  template <> inline nd_double_t const& Entry::view<nd_double_t>() const { return _val(&ndd); }
  template <> inline nd_complex_t const& Entry::view<nd_complex_t>() const { return _val(&ndz); }
}

#endif
