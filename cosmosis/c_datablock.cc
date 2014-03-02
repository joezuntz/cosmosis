#include "datablock.hh"
#include "c_datablock.h"

extern "C"
{
  c_datablock* make_c_datablock(void)
  {
    return new cosmosis::DataBlock();
  }

  DATABLOCK_STATUS destroy_c_datablock(c_datablock* s)
  {
    if (s == nullptr) return DBS_DATABLOCK_NULL;
    delete static_cast<cosmosis::DataBlock*>(s);
    return DBS_SUCCESS;
  }

  DATABLOCK_STATUS
  c_datablock_get_double(c_datablock const* s,
			 const char* section,
			 const char* name,
			 double* val)
  {
    auto p = static_cast<cosmosis::DataBlock const*>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_get_int(c_datablock const* s,
		      const char* section,
		      const char* name,
		      int* val)
  {
    auto p = static_cast<cosmosis::DataBlock const*>(s);
    return p->get_val(section, name, *val);
  }

  DATABLOCK_STATUS
  c_datablock_put_double(c_datablock* s,
			 const char* section,
			 const char* name,
			 double val)
  {
    auto p = static_cast<cosmosis::DataBlock*>(s);
    return p->put_val(section, name, val);
  }

  DATABLOCK_STATUS
  c_datablock_put_int(c_datablock* s,
		      const char* section,
		      const char* name,
		      int val)
  {
    auto p = static_cast<cosmosis::DataBlock*>(s);
    return p->put_val(section, name, val);
  }

} // extern "C"
