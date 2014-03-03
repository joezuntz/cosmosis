#include "datablock.hh"
#include "c_datablock.h"

using cosmosis::DataBlock;


extern "C"
{
  c_datablock* make_c_datablock(void)
  {
    return new cosmosis::DataBlock();
  }

  DATABLOCK_STATUS c_datablock_has_section(c_datablock const* s, const char* name)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    if (name == nullptr) return DBS_NAME_NULL;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    if (p->has_section(name))
      return DBS_SUCCESS;
    else
      return DBS_SECTION_NOT_FOUND;
  }

  int c_datablock_num_sections(c_datablock const* s)
  {
    if (s == nullptr) return -1;
    DataBlock const* p = static_cast<DataBlock const*>(s);
    return p->num_sections();
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
  c_datablock_get_double(c_datablock const* s,
			 const char* section,
			 const char* name,
			 double* val)
  {
    auto p = static_cast<DataBlock const*>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_int(c_datablock const* s,
		      const char* section,
		      const char* name,
		      int* val)
  {
    auto p = static_cast<DataBlock const*>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_put_double(c_datablock* s,
			 const char* section,
			 const char* name,
			 double val)
  {
    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_int(c_datablock* s,
		      const char* section,
		      const char* name,
		      int val)
  {
    auto p = static_cast<DataBlock*>(s);
    return p->put_val(section, name, val);
  }

} // extern "C"
