#include "entry.hh"

using std::string;
using std::vector;

cosmosis::Entry::~Entry()
{
  _destroy_if_managed();
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
    case tag_t::string_array_t:  vs.~vector<std::string>(); break;
    case tag_t::complex_array_t: vz.~vector<complex_t>(); break;
    default:                     break;
    }

}

void cosmosis::Entry::set_int_val(int v)
{
  _destroy_if_managed();
  type_ = tag_t::int_t;
  i = v;
}

void cosmosis::Entry::set_double_val(double v)
{
  _destroy_if_managed();
  type_ = tag_t::double_t;
  d = v;
}

void cosmosis::Entry::set_string_val(string const& v)
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

void cosmosis::Entry::set_complex_val(cosmosis::complex_t v)
{
  _destroy_if_managed();
  type_ = tag_t::complex_t;
  z = v;
}

void cosmosis::Entry::set_int_array(std::vector<int> const& v)
{
  if (type_ == tag_t::int_array_t)
    vi = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::int_array_t;
      new(&vi) std::vector<int>(v);
    }
}

void cosmosis::Entry::set_double_array(std::vector<double> const& v)
{
  if (type_ == tag_t::double_array_t)
    vd = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::double_array_t;
      new(&vd) std::vector<double>(v);
    }
}

void cosmosis::Entry::set_string_array(std::vector<std::string> const& v)
{
  if (type_ == tag_t::string_array_t)
    vs = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::string_array_t;
      new(&vs) std::vector<std::string>(v);
    }
}

void cosmosis::Entry::set_complex_array(std::vector<complex_t> const& v)
{
  if (type_ == tag_t::complex_array_t)
    vz = v;
  else
    {
      _destroy_if_managed();
      type_ = tag_t::complex_array_t;
      new(&vz) std::vector<complex_t>(v);
    }
}


