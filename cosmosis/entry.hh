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
// TODO:
//   1. Check carefully which functions should be qualified 'noexcept'.
//   2. Determine whether strings and vectors should be returned as const&.
//   3. Evaluate whether move c'tor and move assignment should be supported.
//   4. Does is make sense to have a default parameter type?

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
    // explicit Entry(std::vector<int> const& a);
    // explicit Entry(std::vector<double> const& a);
    // explicit Entry(std::vector<std::string> const& a);
    // explicit Entry(std::vector<complex_t> const& a);

    Entry(Entry const& other) = delete;
    Entry& operator=(Entry const& other) = delete;

    ~Entry();

    int int_val() const;
    double double_val() const;
    std::string string_val() const;
    complex_t complex_val() const;
    // std::vector<int> int_array() const;
    // std::vector<double> double_array() const;
    // std::vector<std::string> string_array() const;
    // std::vector<complex_t> complex_array() const;
    
    bool is_int() const;
    bool is_double() const;
    bool is_string() const;
    bool is_complex() const;
    // bool is_int_array() const;
    // bool is_double_array() const;
    // bool is_string_array() const;
    // bool is_complex_array() const;
    
    void set_int_val(int v);
    void set_double_val(double v);
    void set_string_val(std::string const& v);
    void set_complex_val(complex_t v);
    // void set_int_array(std::vector<int> const & a);
    // void set_double_array(std::vector<double> const& a);
    // void set_string_array(std::vector<std::string> const& a);
    // void set_complex_array(std::vector<complex_t> const& a);

  private:
    // tag_t names all the alternatives for what can be stored in the
    // union.
    enum class tag_t { int_t, double_t, string_t, complex_t
	//, int_array_t, double_array_t, string_array_t, complex_array_t 
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

#endif
