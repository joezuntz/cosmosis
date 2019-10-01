#include "entry.hh"
#include "clamp.hh"
#include <limits>
#include <cstdio>

using std::string;
using std::vector;

// We initialize one of the members of the anonymous union in order to
// avoid warnings about the use of unitialized memory.
cosmosis::Entry::Entry(Entry const& e) :
  type_(e.type_),
  i(0)
{
  if      (type_ == enum_for_type<int>()) i = e.i;
  else if (type_ == enum_for_type<bool>()) b = e.b;
  else if (type_ == enum_for_type<double>()) d = e.d;
  else if (type_ == enum_for_type<string>()) emplace(&s, e.s);
  else if (type_ == enum_for_type<complex_t>()) z = e.z;
  else if (type_ == enum_for_type<vint_t>()) emplace(&vi, e.vi);
  else if (type_ == enum_for_type<vdouble_t>()) emplace(&vd, e.vd);
  else if (type_ == enum_for_type<vstring_t>()) emplace(&vs, e.vs);
  else if (type_ == enum_for_type<vcomplex_t>()) emplace(&vz,  e.vz);
  else if (type_ == enum_for_type<nd_int_t>()) emplace(&ndi, e.ndi);
  else if (type_ == enum_for_type<nd_double_t>()) emplace(&ndd, e.ndd);
  else if (type_ == enum_for_type<nd_complex_t>()) emplace(&ndz, e.ndz);
  else throw BadEntry();
}

cosmosis::Entry&
cosmosis::Entry::operator=(cosmosis::Entry const& e)
{
  _destroy_if_managed();
  type_ = e.type_;
  if      (type_ == enum_for_type<int>()) i = e.i;
  else if (type_ == enum_for_type<bool>()) b = e.b;
  else if (type_ == enum_for_type<double>()) d = e.d;
  else if (type_ == enum_for_type<string>()) emplace(&s, e.s);
  else if (type_ == enum_for_type<complex_t>()) z = e.z;
  else if (type_ == enum_for_type<vint_t>()) emplace(&vi, e.vi);
  else if (type_ == enum_for_type<vdouble_t>()) emplace(&vd, e.vd);
  else if (type_ == enum_for_type<vstring_t>()) emplace(&vs, e.vs);
  else if (type_ == enum_for_type<vcomplex_t>()) emplace(&vz,  e.vz);
  else if (type_ == enum_for_type<nd_int_t>()) emplace(&ndi, e.ndi);
  else if (type_ == enum_for_type<nd_double_t>()) emplace(&ndd, e.ndd);
  else if (type_ == enum_for_type<nd_complex_t>()) emplace(&ndz, e.ndz);
  else throw BadEntry();  
  return *this;
}

cosmosis::Entry::~Entry()
{
  _destroy_if_managed();
}

bool
cosmosis::Entry::operator==(Entry const& rhs) const
{
  if (type_ != rhs.type_) return false;
  if (type_ == enum_for_type<int>()) return i == rhs.i;
  else if (type_ == enum_for_type<bool>()) return b == rhs.b;
  else if (type_ == enum_for_type<double>()) return d == rhs.d;
  else if (type_ == enum_for_type<string>()) return s == rhs.s;
  else if (type_ == enum_for_type<complex_t>()) return z == rhs.z;
  else if (type_ == enum_for_type<vint_t>()) return vi == rhs.vi;
  else if (type_ == enum_for_type<vdouble_t>()) return vd == rhs.vd;
  else if (type_ == enum_for_type<vstring_t>()) return vs == rhs.vs;
  else if (type_ == enum_for_type<vcomplex_t>()) return vz == rhs.vz;
  else if (type_ == enum_for_type<nd_int_t>()) return ndi == rhs.ndi;
  else if (type_ == enum_for_type<nd_double_t>()) return ndd == rhs.ndd;
  else if (type_ == enum_for_type<nd_complex_t>()) return ndz == rhs.ndz;
  else throw BadEntry();
}

// Each 'set' function must check for all possible types with
// user-defined c'tor, for proper destruction of the old value.
//
// Set functions for types with user-defined c'tors must use placement
// new to construct the (copied) value in-place.

// Call the appropriate in-place destructor, if we're carrying a memory-managed
// type.

void cosmosis::Entry::_destroy_if_managed() {
  if      (type_ == enum_for_type<string>()) s.~string();
  else if (type_ == enum_for_type<vint_t>()) vi.~vector<int>();
  else if (type_ == enum_for_type<vdouble_t>()) vd.~vector<double>();
  else if (type_ == enum_for_type<vstring_t>()) vs.~vector<string>();
  else if (type_ == enum_for_type<vcomplex_t>()) vz.~vector<complex_t>();
  else if (type_ == enum_for_type<nd_int_t>()) ndi.~ndarray<int>();
  else if (type_ == enum_for_type<nd_double_t>()) ndd.~ndarray<double>();
  else if (type_ == enum_for_type<nd_complex_t>()) ndz.~ndarray<complex_t>();
}

template <class V>
int clamped_size(V const& v)
{
  return cosmosis::clamp(v.size());
}

// Should size() deal with ndarray values? What would it make sense to
// return?
int cosmosis::Entry::size() const
{
  if      (type_ == enum_for_type<vint_t>())     return clamped_size(vi);
  else if (type_ == enum_for_type<vdouble_t>())  return clamped_size(vd);
  else if (type_ == enum_for_type<vstring_t>())  return clamped_size(vs);
  else if (type_ == enum_for_type<vcomplex_t>()) return clamped_size(vz);
  else return -1;  
}

void cosmosis::Entry::set_val(int v) { _set(v, i); }
void cosmosis::Entry::set_val(bool v) { _set(v, b); }
void cosmosis::Entry::set_val(double v) { _set(v, d); }
void cosmosis::Entry::set_val(const char * v) { _vset(string(v), s); }
void cosmosis::Entry::set_val(string const& v) { _vset(v, s); }
void cosmosis::Entry::set_val(cosmosis::complex_t v) { _set(v, z); }
void cosmosis::Entry::set_val(vector<int> const& v) { _vset(v, vi); }
void cosmosis::Entry::set_val(vector<double> const& v) { _vset(v, vd); }
void cosmosis::Entry::set_val(vector<string> const& v) { _vset(v, vs); }
void cosmosis::Entry::set_val(vector<complex_t> const& v) { _vset(v, vz); }

void cosmosis::Entry::set_val(nd_int_t const& v) { _vset(v, ndi); }
void cosmosis::Entry::set_val(nd_double_t const& v) { _vset(v, ndd); }
void cosmosis::Entry::set_val(nd_complex_t const& v) { _vset(v, ndz); }
