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

  class Entry
  {
  public:
    struct BadEntry { }; // used for exceptions.

    explicit Entry(int v);
    explicit Entry(double v);
    explicit Entry(std::string v);
    explicit Entry(complex_t v);
    explicit Entry(std::vector<int> const& a);
    explicit Entry(std::vector<double> const& a);
    explicit Entry(std::vector<std::string> const& a);
    explicit Entry(std::vector<complex_t> const& a);

    Entry(Entry const& other);
    Entry& operator=(Entry const& other);

    ~Entry();

    int int_val() const;
    double double_val() const;
    std::string string_val() const;
    complex_t complex_val() const;
    std::vector<int> int_array() const;
    std::vector<double> double_array() const;
    std::vector<std::string> string_array() const;
    std::vector<complex_t> complex_array() const;

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
    void set_int_array(std::vector<int> const & a);
    void set_double_array(std::vector<double> const& a);
    void set_string_array(std::vector<std::string> const& a);
    void set_complex_array(std::vector<complex_t> const& a);

  private:
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
      std::vector<int> vi;
      std::vector<double> vd;
      std::vector<std::string> vs;
      std::vector<complex_t> vz;
    }; // union

    void _destroy_if_managed();

  }; // class Entry
} // namespace cosmosis

// Implementation of member functions.


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
cosmosis::Entry::Entry(std::vector<int> const& v) :
  type_(tag_t::int_array_t), vi(v)
{}

inline
cosmosis::Entry::Entry(std::vector<double> const& v) :
  type_(tag_t::double_array_t), vd(v)
{}

inline
cosmosis::Entry::Entry(std::vector<std::string> const& v) :
  type_(tag_t::string_array_t), vs(v)
{}

inline
cosmosis::Entry::Entry(std::vector<complex_t> const& v) :
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
std::vector<int> cosmosis::Entry::int_array() const
{
  if (type_ != tag_t::int_array_t) throw BadEntry();
  return vi;
}

inline
std::vector<double> cosmosis::Entry::double_array() const
{
  if (type_ != tag_t::double_array_t) throw BadEntry();
  return vd;
}

inline
std::vector<std::string> cosmosis::Entry::string_array() const
{
  if (type_ != tag_t::string_array_t) throw BadEntry();
  return vs;
}

inline
std::vector<cosmosis::complex_t> cosmosis::Entry::complex_array() const
{
  if (type_ != tag_t::complex_array_t) throw BadEntry();
  return vz;
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
