cimport numpy as nm
import numpy as nm
nm.import_array()
cimport stdlib as stdlib
cimport stdio as stdio
cdef extern from "string.h" nogil:
    void *memcpy  (void *TO, void *FROM, size_t SIZE)

cdef extern from "errorlist.h":
  ctypedef struct error:
    pass
    
  void stringError(char* str, error *err)
  int getErrorValue(error* err)
  int isError(error *err) 
  void purgeError(error **err) 
  void printError(void* flog,error* err)

class CError(Exception):
  def __init__(self,val,str):
    self.val=val
    self.comment=str
  def __str__(self):
    return self.comment.strip()

cdef doError(error **err):
  cdef char estr[10000]
  if (isError(err[0])):
    stringError(estr,err[0])
    er=CError(getErrorValue(err[0]),estr)
    purgeError(err)
    
    return er
  return None

cdef extern from "clik.h":
  ctypedef char parname[256]
  ctypedef void clik_lensing_object
  
  clik_lensing_object* clik_lensing_init(char* hdffilepath, error **err)
  int clik_lensing_get_extra_parameter_names(clik_lensing_object* self, parname **names, error **err)
  int clik_lensing_get_lmax(clik_lensing_object *self,error **err)
  double clik_lensing_compute(clik_lensing_object* self, double* cl_and_pars,error **err)
  void clik_lensing_cleanup(clik_lensing_object** pself)
  double* clik_lensing_cltt_fid(clik_lensing_object* clikid, error **_err)
  double* clik_lensing_clpp_fid(clik_lensing_object* clikid, error **_err)
  int clik_try_lensing(char *fpath,error **_err)

cdef class clik_lensing:
  cdef clik_lensing_object* celf
  cdef error *_err,**err
  cdef int ndim,pdim  
  
  def __init__(self,filename):
    self.celf=NULL
    self._err = NULL
    self.err = &self._err
    
    self.celf = clik_lensing_init(filename,self.err)
    er=doError(self.err)
    if er:
      raise er
    lmax = self.lmax
    extra = self.extra_parameter_names
    nn = (lmax+1)*2
    self.ndim = nn + len(extra)
    self.pdim = lmax+1+len(extra)

  def __call__(self,pars):
    cdef int lmax

    lmax = self.lmax
    pars_2d = nm.atleast_2d(pars)
    if pars_2d.shape[1] not in (self.ndim,self.pdim) :
      raise Exception("Bad shape (expecting (-1,%d) or (-1,%d) got (%d,%d))"%(self.ndim,self.pdim,pars_2d.shape[0],pars_2d.shape[1]))
    res = nm.zeros(pars_2d.shape[0],dtype=nm.double)
    i=0
    for apars in pars_2d:
      if apars.shape[0] == self.pdim:
        qpars = nm.concatenate((apars[:lmax+1],self.get_cltt_fid(),apars[lmax+1:]))
        pars_proxy=nm.PyArray_ContiguousFromAny(qpars,nm.NPY_DOUBLE,1,1)
      else:
        pars_proxy=nm.PyArray_ContiguousFromAny(apars,nm.NPY_DOUBLE,1,1)
      res[i] = clik_lensing_compute(self.celf,<double*> nm.PyArray_DATA(pars_proxy),self.err)
      er=doError(self.err)
      if er:
        raise er
      i+=1
    return res
    
  def __dealloc__(self):
    if self.celf!=NULL:
      clik_lensing_cleanup(&(self.celf))
        
  def get_lmax(self):
    cdef int lmax
    lmax = clik_lensing_get_lmax(self.celf, self.err)
    er=doError(self.err)
    if er:
      raise er
    return lmax
    
  property lmax:
    def __get__(self):
      return self.get_lmax()
  
  def get_extra_parameter_names(self):
    cdef parname *names
    
    n_names = clik_lensing_get_extra_parameter_names(self.celf, &names, self.err)    
    res = ["%s"%names[i] for i in range(n_names)]
    stdlib.free(names)
    return tuple(res)

    
  property extra_parameter_names:
    def __get__(self):
      return self.get_extra_parameter_names()

  def get_cltt_fid(self):
    cdef double *cltt
    cdef int lmax

    cltt = clik_lensing_cltt_fid(self.celf,self.err)
    er=doError(self.err)
    if er:
      raise er
    lmax = self.lmax
    rltt = nm.zeros(lmax+1,dtype=nm.double)
    memcpy(<void*> nm.PyArray_DATA(rltt), cltt,sizeof(double)*(lmax+1))
    stdlib.free(cltt)
    return rltt

  def get_clpp_fid(self):
    cdef double *cltt
    cdef int lmax

    cltt = clik_lensing_clpp_fid(self.celf,self.err)
    er=doError(self.err)
    if er:
      raise er
    lmax = self.lmax
    rltt = nm.zeros(lmax+1,dtype=nm.double)
    memcpy(<void*> nm.PyArray_DATA(rltt), cltt,sizeof(double)*(lmax+1))
    stdlib.free(cltt)
    return rltt

def try_lensing(fl):
  cdef error *_err,**err
  _err = NULL
  err = &_err
    
  r = clik_try_lensing(fl,err)
  er=doError(err)
  if er:
    raise er
    
  return bool(r==1)

