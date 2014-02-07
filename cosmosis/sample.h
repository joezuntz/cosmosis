#ifndef COSMOSIS_SAMPLE_H
#define COSMOSIS_SAMPLE_H

#ifdef __cplusplus
#include <string>
#include <map>

//----------------------------------------------------------------------
// This class demostrates some ways to implement the DataPackage
// functionality in C++, and provides a C binding to the same
// functionality.
//
// It has some redundancies, to show different options for the same
// functionality. In the real class, such redundancies would be
// removed.
//
// The DataBlock interface in C and Fortran do not provide direct
// access to the memory controlled by the DataBlock; they provide the
// user with a copy of the data, copied to a memory location of the
// user's specification. For any array obtained from the DataBlock,
// the C or Fortran user must deallocate the array when he is done
// with it. The deallocation does not need to be done for arrays that
// are on the stack, into which DataBlock copies the data.
//
// In a "debugging state" every access should write a message to some
// log.
//
//----------------------------------------------------------------------

namespace cosmosis
{
  class Sample
  {
  public:
    // All memory management functions are compiler generated.
    
    // getX functions return false if the given name is not found.
    bool getDouble(std::string const& name, double& val) const noexcept;

    // get_X functions throw if the given name is not found. 
    double get_double(std::string const& name) const;

    void putDouble(std::string const& name, double val);
    void replaceDouble(std::string const& name, double val);
    
  private:
    std::map<std::string, double> values_;  
  };
}

extern "C"
{
#endif

  typedef void c_sample;
  c_sample* make_c_sample(void);

  bool c_sample_has_section(c_sample const* s, const char* name);
  bool c_sample_has_value(c_sample const* s, const char* section, const char* name);
  int c_sample_num_sections(....);
  int c_sample_get_section_name(..., int isection);  

  void destroy_c_sample(c_sample* s);

  /* 
     Return 0 if a double named 'name' is found. We do no conversions of type.
     1: section not found.
     2: name not found
     3: wrong type
     4: memory allocation failure.
     5: section name is null
     6: name is null, section is not null
     7: val is null.
     8: s is null.

     If the return status is nonzero, the value written into 'val' is not defined.
     If the return status is nonzero, the value written into 'val' is NaN.
  */
  int  c_sample_get_double(c_sample const* s, const char* name, double* val);

  /* Only scalars have default in the C and Fortran interfaces. */
  int  c_sample_get_double_default(c_sample const* s, const char* name, double* val, double dflt);


  /* 
     Return 0 if the put worked, and nonzero to indicate failure.
     1: name already exists
     2: memory allocation failure
  */
  int c_sample_put_double(c_sample* s, const char* name, double val);

  /*
     Return 0 if the put worked, and nonzero to indicate failure.
     1: name does not already exist.
     2: memory allocation failure.
     3: replace of wrong type.
   */
  int c_sample_replace_double(c_sample* s, const char* name, double val);

  /* Return 0 if the put worked, and nonzero to indicate failure */
  int c_sample_get_double_array_1d(c_sample const* s, const char* name,
				   double** array,
				   int* size);

  int c_sample_get_double_array_1d_preallocated(c_sample const* s, const char* name,
						double* array,
						int* size,
						int maxsize);

  void c_sample_put_double_array_1d(c_sample* s, const char* name,
				    double* array, int sz);

  

#ifdef __cplusplus
}
#endif

#endif
