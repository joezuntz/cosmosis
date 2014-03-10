#include <assert.h>
#include <complex.h>
#include <string.h>

#include "datablock.hh"
#include "section.hh"
#include "entry.hh"
#include "c_datablock.h"

using cosmosis::DataBlock;
using cosmosis::Section;
using cosmosis::Entry;
using cosmosis::complex_t;
using std::string;
using std::vector;

extern "C"
{
  c_datablock* make_c_datablock(void)
  {
    return new cosmosis::DataBlock();
  }

  _Bool c_datablock_has_section(c_datablock const* s, const char* name)
  {
    if (s == nullptr || name == nullptr) return false;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->has_section(name);
  }

  int c_datablock_num_sections(c_datablock const* s)
  {
    if (s == nullptr) return -1;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->num_sections();
  }

  DATABLOCK_STATUS c_datablock_has_value(c_datablock const* s,
                                         const char* section,
                                         const char* name)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->has_val(section, name);
  }

  const char* c_datablock_get_section_name(c_datablock const* s, int i)
  {
    if (i < 0) return nullptr;
    auto n = static_cast<size_t>(i);
    DataBlock const* p = static_cast<DataBlock const*>(s);
    if (n >= p->num_sections()) return nullptr;
    return p->section_name(n).c_str();
  }

  DATABLOCK_STATUS destroy_c_datablock(c_datablock* s)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    DataBlock* p = static_cast<DataBlock*>(s);
    // The call to clear() is not really necessary, but to aid in
    // debugging incorrect use of the C interface (especially to help
    // detect premature calls to destroy_c_datablock), it seems
    // useful.
    p->clear();
    delete p;
    return DBS_SUCCESS;
  }


  DATABLOCK_STATUS
  c_datablock_get_int(c_datablock const* s,
		      const char* section,
		      const char* name,
		      int* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_double(c_datablock const* s,
			 const char* section,
			 const char* name,
			 double* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_complex(c_datablock const* s,
			  const char* section,
			  const char* name,
			  double _Complex* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    complex_t z;
    auto rc = p->get_val(section, name, z);
    // C11 provides a function macro to create a double _Complex from
    // real and imaginary parts, but we don't require a C11-compliant
    // compiler. I would expect
    //
    //    *val = z.real() + z.imag() * _Complex_I;
    //
    // to work, but it produces a compilation failure with
    // GCC 4.8.2.  The cast below is unattractive, but works because
    // C++11 promises layout compatibility between
    // std::complex<double> and double[2], and C makes the similar
    // guarantee for double _Complex.
    if (rc == DBS_SUCCESS) *val = * reinterpret_cast<double _Complex*>(&z);
    return rc;
  }

  DATABLOCK_STATUS
  c_datablock_get_string(c_datablock const* s,
                         const char* section,
                         const char* name,
                         char**  val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    string tmp;
    auto rc = p->get_val(section, name, tmp);
    if (rc != DBS_SUCCESS) return rc;
    *val = strdup(tmp.c_str());
    if (*val == nullptr) return DBS_MEMORY_ALLOC_FAILURE;
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_int_array_1d(c_datablock const* s,
                               const char* section,
                               const char* name,
                               int** val,
                               int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    try {
      vector<int> const& r = p->view<vector<int>>(section, name);
      *val = static_cast<int*>(malloc(r.size() * sizeof(int)));
      if (*val ==nullptr) return DBS_MEMORY_ALLOC_FAILURE;
      std::copy(r.cbegin(), r.cend(), *val);
      *sz = r.size();
      return DBS_SUCCESS;
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_LOGIC_ERROR;
  }

  DATABLOCK_STATUS
  c_datablock_get_double_array_1d(c_datablock const* s,
                               const char* section,
                               const char* name,
                               double** val,
                               int* sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    try {
      vector<double> const& r = p->view<vector<double>>(section, name);
      *val = static_cast<double*>(malloc(r.size() * sizeof(double)));
      if (*val ==nullptr) return DBS_MEMORY_ALLOC_FAILURE;
      std::copy(r.cbegin(), r.cend(), *val);
      *sz = r.size();
      return DBS_SUCCESS;
    }
    catch (DataBlock::BadDataBlockAccess const&) { return DBS_SECTION_NOT_FOUND; }
    catch (Section::BadSectionAccess const&) { return DBS_NAME_NOT_FOUND; }
    catch (Entry::BadEntry const&) { return DBS_WRONG_VALUE_TYPE; }
    catch (...) { return DBS_LOGIC_ERROR; }
    return DBS_LOGIC_ERROR;
  }

  DATABLOCK_STATUS
  c_datablock_get_int_array_1d_preallocated(c_datablock const* s,
                                            const char* section,
                                            const char* name,
                                            int* val,
                                            int* sz,
                                            int maxsize)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz == nullptr) return DBS_SIZE_NULL;

    auto p = static_cast<DataBlock const*>(s);
    vector<int> const& r = p->view<vector<int>>(section, name);
    if (r.size() > static_cast<size_t>(maxsize)) return DBS_SIZE_INSUFFICIENT;
    *sz = r.size();
    std::copy(r.cbegin(), r.cend(), val);
    // If we are asked to clear out the remainder of the input buffer,
    // the following line should be used.

    //
    // std::fill(val + *sz, val+maxsize, 0);
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_put_int(c_datablock* s,
		      const char* section,
		      const char* name,
		      int val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_double(c_datablock* s,
			 const char* section,
			 const char* name,
			 double val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_complex(c_datablock* s,
			  const char* section,
			  const char* name,
			  double _Complex val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    complex_t z(val);
    return p->put_val(section, name, z);
  }

  DATABLOCK_STATUS
  c_datablock_put_string(c_datablock* s,
			 const char* section,
			 const char* name,
			 const char* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, string(val));
  }

  DATABLOCK_STATUS
  c_datablock_put_int_array_1d(c_datablock* s,
                               const char* section,
                               const char* name,
                               int const*  val,
                               int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;
    if (sz < 0) return DBS_SIZE_NEGATIVE;

    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, vector<int>(val, val+sz));
  }

  DATABLOCK_STATUS
  c_datablock_replace_int(c_datablock* s,
			  const char* section,
			  const char* name,
			  int val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_replace_double(c_datablock* s,
			     const char* section,
			     const char* name,
			     double val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_replace_complex(c_datablock* s,
			     const char* section,
			     const char* name,
			     double _Complex val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;

    auto p = static_cast<DataBlock*>(s);
    std::complex<double> z { val };
    return p->replace_val(section, name, z);
  }

  DATABLOCK_STATUS
  c_datablock_replace_string(c_datablock* s,
			     const char* section,
			     const char* name,
			     const char* val)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == NULL) return DBS_VALUE_NULL;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, string(val));
  }

  DATABLOCK_STATUS
  c_datablock_replace_int_array_id(c_datablock* s,
                                   const char* section,
                                   const char* name,
                                   int const* val,
                                   int sz)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (section == nullptr) return DBS_SECTION_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    if (val == nullptr) return DBS_VALUE_NULL;
    if (sz  < 0) return DBS_SIZE_NEGATIVE;

    auto p = static_cast<DataBlock*>(s);
    return p->replace_val(section, name, vector<int>(val, val+sz));
  }

} // extern "C"
