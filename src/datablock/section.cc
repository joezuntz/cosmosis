#include "section.hh"

using std::string;

bool
cosmosis::Section::has_val(string const& name) const
{
  return vals_.find(name) != vals_.end();
}

int
cosmosis::Section::number_values() const
{
  return vals_.size();
}

int
cosmosis::Section::get_size(string const& name) const
{
  auto ival = vals_.find(name);
  if (ival == vals_.end()) return -1;
  return ival->second.size();
}


std::string const& cosmosis::Section::value_name(std::size_t i) const
{
  if (i >= (std::size_t) number_values()) throw BadSectionAccess();
  auto isec = vals_.begin();
  std::advance(isec, i);
  return isec->first;

}

DATABLOCK_STATUS 
cosmosis::Section::get_type(std::string const&name, datablock_type_t &t) const
{
  auto ival = vals_.find(name);
  // Find the right entry
  // If not found, use unkown
  if (ival == vals_.end()) {
	t = DBT_UNKNOWN;
  	return DBS_NAME_NOT_FOUND;
  }
  if      (ival->second.is<int>())        t = DBT_INT;
  else if (ival->second.is<double>())     t = DBT_DOUBLE;
  else if (ival->second.is<complex_t>())  t = DBT_COMPLEX;
  else if (ival->second.is<string>())     t = DBT_STRING;
  else if (ival->second.is<string>())     t = DBT_STRING;
  else if (ival->second.is<vint_t>())     t = DBT_INT1D;
  else if (ival->second.is<vdouble_t>())  t = DBT_DOUBLE1D;
  else if (ival->second.is<vcomplex_t>()) t = DBT_COMPLEX1D;
  else if (ival->second.is<vstring_t>())  t = DBT_STRING1D;
  // These are not written yet - add them once types are 
  // defined:
	  // DBT_INT2D,
	  // DBT_DOUBLE2D,
	  // DBT_COMPLEX2D,
	  // DBT_STRING2D,
  else{
  	// We have failed to identify the type!  Perhaps 
  	// this is because one of the 2D functions is being tested
  	// and we did not update this piece of code yet!
	//  Since this should not really
  	// be possible to happen we return a logic error
  	t = DBT_UNKNOWN;
  	return DBS_LOGIC_ERROR;
  }

  return DBS_SUCCESS;



}
