#ifndef COSMOSIS_ENTRY_HH
#define COSMOSIS_ENTRY_HH

#include <string>
#include <complex>
#include <vector>

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
//  The 'is_<type>' functions are present mostly so that the C interface
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
  
  class Entry
  {
  public:
    struct BadEntry { }; // used for exceptions.

    Entry();
    explicit Entry(int v);
    explicit Entry(double v);
    explicit Entry(std::string v);
    explicit Entry(complex_t v);
    explicit Entry(vint_t const& a);
    explicit Entry(vdouble_t const& a);
    explicit Entry(vstring_t const& a);
    explicit Entry(vcomplex_t const& a);

    Entry(Entry const& other);
    Entry& operator=(Entry const& other) = delete;

    ~Entry();

    bool operator==(Entry const& other) const;

    // Return true if the Entry is currently carrying a value of type T.
    template <class T> bool is() const;

    // If the Entry is carrying a value of type T, return
    // it. Otherwise throw a BadEntry exception.
    template <class T> T val() const;

    // Replace the existing value (of whatever type) with the given
    // value.
    void set_val(int v);
    void set_val(double v);
    void set_val(std::string const& v);
    void set_val(complex_t v);
    void set_val(vint_t const& v);
    void set_val(vdouble_t const& v);
    void set_val(vstring_t const& v);
    void set_val(vcomplex_t const& v);

  private:

    int int_val() const;
    double double_val() const;
    std::string string_val() const;
    complex_t complex_val() const;
    vint_t int_array() const;
    vdouble_t double_array() const;
    vstring_t string_array() const;
    vcomplex_t complex_array() const;

    bool is_int() const;
    bool is_double() const;
    bool is_string() const;
    bool is_complex() const;
    bool is_int_array() const;
    bool is_double_array() const;
    bool is_string_array() const;
    bool is_complex_array() const;

    void set_int_val(int v);
    void set_double_val(double v);
    void set_string_val(std::string const& v);
    void set_complex_val(complex_t v);
    void set_int_array(vint_t const & a);
    void set_double_array(vdouble_t const& a);
    void set_string_array(vstring_t const& a);
    void set_complex_array(vcomplex_t const& a);

    // tag_t names all the alternatives for what can be stored in the
    // union.
    enum class tag_t { int_t, double_t, string_t, complex_t
        , int_array_t, double_array_t, string_array_t, complex_array_t

        };
    tag_t type_;

    // the anonymous union contains the value.
    union
    {
      int i;
      double d;
      std::string s;
      complex_t z;
      vint_t vi;
      vdouble_t vd;
      vstring_t vs;
      vcomplex_t vz;
    }; // union

    void _destroy_if_managed();

  }; // class Entry
} // namespace cosmosis

// Implementation of member functions.

inline
cosmosis::Entry::Entry() :
  type_(tag_t::double_t), d(0.0)
{ }

inline
cosmosis::Entry::Entry(int v) :
  type_(tag_t::int_t), i(v)
{}

inline
cosmosis::Entry::Entry(double v) :
  type_(tag_t::double_t), d(v)
{}

inline
cosmosis::Entry::Entry(std::string v) :
  type_(tag_t::string_t), s(v)
{}

inline
cosmosis::Entry::Entry(complex_t v) :
  type_(tag_t::complex_t), z(v)
{}

inline
cosmosis::Entry::Entry(vint_t const& v) :
  type_(tag_t::int_array_t), vi(v)
{}

inline
cosmosis::Entry::Entry(vdouble_t const& v) :
  type_(tag_t::double_array_t), vd(v)
{}

inline
cosmosis::Entry::Entry(vstring_t const& v) :
  type_(tag_t::string_array_t), vs(v)
{}

inline
cosmosis::Entry::Entry(vcomplex_t const& v) :
  type_(tag_t::complex_array_t), vz(v)
{}

inline
int cosmosis::Entry::int_val() const
{
  if (type_ != tag_t::int_t) throw BadEntry();
  return i;
}

inline
double cosmosis::Entry::double_val() const
{
  if (type_ != tag_t::double_t) throw BadEntry();
  return d;
}

inline
std::string cosmosis::Entry::string_val() const
{
  if (type_ != tag_t::string_t) throw BadEntry();
  return s;
}

inline
cosmosis::complex_t cosmosis::Entry::complex_val() const
{
  if (type_ != tag_t::complex_t) throw BadEntry();
  return z;
}

inline
cosmosis::vint_t cosmosis::Entry::int_array() const
{
  if (type_ != tag_t::int_array_t) throw BadEntry();
  return vi;
}

inline
cosmosis::vdouble_t cosmosis::Entry::double_array() const
{
  if (type_ != tag_t::double_array_t) throw BadEntry();
  return vd;
}

inline
cosmosis::vstring_t cosmosis::Entry::string_array() const
{
  if (type_ != tag_t::string_array_t) throw BadEntry();
  return vs;
}

inline
cosmosis::vcomplex_t cosmosis::Entry::complex_array() const
{
  if (type_ != tag_t::complex_array_t) throw BadEntry();
  return vz;
}

namespace cosmosis
{
  template <> inline bool Entry::is<int>() const { return is_int(); }
  template <> inline int Entry::val<int>() const { return int_val(); }

  template <> inline bool Entry::is<double>() const { return is_double(); }
  template <> inline double Entry::val<double>() const { return double_val(); }
  
  template <> inline bool Entry::is<std::string>() const { return is_string(); }
  template <> inline std::string Entry::val<std::string>() const { return string_val(); }

  template <> inline bool Entry::is<complex_t>() const { return is_complex(); }
  template <> inline complex_t Entry::val<complex_t>() const { return complex_val(); }

  template <> inline bool Entry::is<vint_t>() const { return is_int_array(); }
  template <> inline vint_t Entry::val<vint_t>() const { return int_array(); }

  template <> inline bool Entry::is<vdouble_t>() const { return is_double_array(); }
  template <> inline vdouble_t Entry::val<vdouble_t>() const { return double_array(); }

  template <> inline bool Entry::is<vstring_t>() const { return is_string_array(); }
  template <> inline vstring_t Entry::val<vstring_t>() const { return string_array(); }

  template <> inline bool Entry::is<vcomplex_t>() const { return is_complex_array(); }
  template <> inline vcomplex_t Entry::val<vcomplex_t>() const { return complex_array(); }
}


inline
bool cosmosis::Entry::is_int() const
{
  return (type_ == tag_t::int_t);
}

inline
bool cosmosis::Entry::is_double() const
{
  return (type_ == tag_t::double_t);
}

inline
bool cosmosis::Entry::is_string() const
{
  return (type_ == tag_t::string_t);
}

inline
bool cosmosis::Entry::is_complex() const
{
  return (type_ == tag_t::complex_t);
}

inline
bool cosmosis::Entry::is_int_array() const
{
  return (type_ == tag_t::int_array_t);
}

inline
bool cosmosis::Entry::is_double_array() const
{
  return (type_ == tag_t::double_array_t);
}

inline
bool cosmosis::Entry::is_string_array() const
{
  return (type_ == tag_t::string_array_t);
}

inline
bool cosmosis::Entry::is_complex_array() const
{
  return (type_ == tag_t::complex_array_t);
}

#endif
