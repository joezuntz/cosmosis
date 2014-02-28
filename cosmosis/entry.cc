#include "entry.hh"

using std::string;
using std::vector;

// This seems inefficient, but we must first set the data to an
// innocuous value, so that the 'set' functions we call don't cause
// havoc based on our current state.
cosmosis::Entry::Entry(Entry const& e) :
  type_(e.type_)
{
  switch (type_)
    {
    case tag_t::int_t:            i = e.i; break;
    case tag_t::double_t:         d = e.d; break;
    case tag_t::string_t:         new(&s) string(e.s); break;
    case tag_t::complex_t:        z = e.z; break;
    case tag_t::int_array_t:      new(&vi) vector<int>(e.vi); break;
    case tag_t::double_array_t:   new(&vd) vector<double>(e.vd); break;
    case tag_t::string_array_t:   new(&vs) vector<string>(e.vs); break;
    case tag_t::complex_array_t:  new(&vz) vector<complex_t>(e.vz); break;
    default: throw BadEntry();
    }
}

cosmosis::Entry::~Entry()
{
  _destroy_if_managed();
}

bool
cosmosis::Entry::operator==(Entry const& rhs) const
{
  if (type_ != rhs.type_) return false;
  switch (type_)
    {
    case tag_t::int_t:            return i == rhs.i;
    case tag_t::double_t:         return d == rhs.d;
    case tag_t::string_t:         return s == rhs.s;
    case tag_t::complex_t:        return z == rhs.z;
    case tag_t::int_array_t:     return vi == rhs.vi;
    case tag_t::double_array_t:  return vd == rhs.vd;
    case tag_t::string_array_t:  return vs == rhs.vs;
    case tag_t::complex_array_t: return vz == rhs.vz;
    default: throw BadEntry();
    }

}

// Each 'set' function must check for all possible types with
// user-defined c'tor, for proper destruction of the old value.
//
// Set functions for types with user-defined c'tors must use placement
// new to construct the (copied) value in-place.

// Call the appropriate in-place destructor, if we're carrying a memory-managed
// type.

void cosmosis::Entry::_destroy_if_managed() {
  switch (type_)
    {
    case tag_t::string_t:        s.~string(); break;
    case tag_t::int_array_t:     vi.~vector<int>(); break;
    case tag_t::double_array_t:  vd.~vector<double>(); break;
    case tag_t::string_array_t:  vs.~vector<string>(); break;
    case tag_t::complex_array_t: vz.~vector<complex_t>(); break;
    default:                     break;
    }
}

void cosmosis::Entry::set_val(int v)
{
  _destroy_if_managed();
  type_ = tag_t::int_t;
  i = v;
}

void cosmosis::Entry::set_val(double v)
{
  _destroy_if_managed();
  type_ = tag_t::double_t;
  d = v;
}

void cosmosis::Entry::set_val(string const& v)
{
  if (type_ == tag_t::string_t)
    s = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::string_t;
      new(&s) string(v);
    }
}

void cosmosis::Entry::set_val(cosmosis::complex_t v)
{
  _destroy_if_managed();
  type_ = tag_t::complex_t;
  z = v;
}

void cosmosis::Entry::set_val(vector<int> const& v)
{
  if (type_ == tag_t::int_array_t)
    vi = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::int_array_t;
      new(&vi) vector<int>(v);
    }
}

void cosmosis::Entry::set_val(vector<double> const& v)
{
  if (type_ == tag_t::double_array_t)
    vd = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::double_array_t;
      new(&vd) vector<double>(v);
    }
}

void cosmosis::Entry::set_val(vector<string> const& v)
{
  if (type_ == tag_t::string_array_t)
    vs = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::string_array_t;
      new(&vs) vector<string>(v);
    }
}

void cosmosis::Entry::set_val(vector<complex_t> const& v)
{
  if (type_ == tag_t::complex_array_t)
    vz = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::complex_array_t;
      new(&vz) vector<complex_t>(v);
    }
}
