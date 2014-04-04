cdef extern from "errorlist.h":
  ctypedef struct error:
    pass
    
cdef object doError(error **err)

cdef extern from "clik_parametric.h":
  ctypedef struct c_parametric "parametric":
    int lmin,lmax,ndet,nfreq,nvar

ctypedef c_parametric* (*simple_init)(int , double *, int , char** , char **, int , char **, int , int , error **)
ctypedef c_parametric* (*template_init)(int , double *, int , char** , char **, int , char **, int , int , double*, error **)

cdef class parametric:
  cdef c_parametric* celf
  cdef int nell
  cdef readonly object varpar,parvalues
  cdef void* initfunc
  cdef object rename,emaner

cdef class parametric_template(parametric):
  cdef object template_name
  cdef object plugin_name