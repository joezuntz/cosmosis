#include "entry.hh"

using std::string;
using std::vector;

cosmosis::Entry::~Entry()
{
  if (type_ == tag_t::string_t) s.~string();
}

// Each 'set' function must check for all possible types with
// user-defined c'tor, for proper destruction of the old value.
//
// Set functions for types with user-defined c'tors must use placement
// new to construct the (copied) value in-place.

void cosmosis::Entry::set_int_val(int v)
{
  if (type_ == tag_t::string_t) s.~string();
  type_ = tag_t::int_t;
  i = v;
}

void cosmosis::Entry::set_double_val(double v)
{
  if (type_ == tag_t::string_t) s.~string();
  type_ = tag_t::double_t;
  d = v;
}

void cosmosis::Entry::set_string_val(string const& v)
{
  if (type_ == tag_t::string_t)
    s = v;
  else
    {
      new(&s) string(v);
      type_ = tag_t::string_t;
    }
}

void cosmosis::Entry::set_complex_val(cosmosis::complex_t v)
{
  if (type_ == tag_t::string_t) s.~string();
  type_ = tag_t::complex_t;
  z = v;
}
