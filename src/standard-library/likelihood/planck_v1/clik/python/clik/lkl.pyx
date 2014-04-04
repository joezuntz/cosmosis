cimport numpy as nm
import numpy as nm
nm.import_array()
cimport stdlib as stdlib
cimport stdio as stdio

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
  ctypedef void clik_object
  
  clik_object* clik_init(char* hdffilepath, error **err)
  void clik_get_has_cl(clik_object *self, int has_cl[6],error **err)
  int clik_get_extra_parameter_names(clik_object* self, parname **names, error **err)
  void clik_get_lmax(clik_object *self, int lmax[6],error **err)
  double clik_compute(clik_object* self, double* cl_and_pars,error **err)
  void clik_cleanup(clik_object** pself)
  int clik_get_extra_parameter_names_by_lkl(clik_object* clikid, int ilkl,parname **names, error **_err)
  char* clik_get_version(clik_object *clikid,error **_err)

cdef class clik:
  cdef clik_object* celf
  cdef error *_err,**err
  cdef int ndim  
  
  def version(self):
    cdef char* ver_str

    ver_str = clik_get_version(self.celf,self.err)
    er=doError(self.err)
    if er:
      raise er
    pyver = ver_str
    pyver = pyver + ""
    stdlib.free(ver_str)
    return pyver
    
  def __init__(self,filename):
    self.celf=NULL
    self._err = NULL
    self.err = &self._err
    
    self.celf = clik_init(filename,self.err)
    er=doError(self.err)
    if er:
      raise er
    lmax = self.lmax
    extra = self.extra_parameter_names
    nn = nm.sum(nm.array(lmax)+1)
    self.ndim = nn + len(extra)
    
  def __call__(self,pars):
    pars_2d = nm.atleast_2d(pars)
    if pars_2d.shape[1]!=self.ndim:
      raise Exception("Bad shape (expecting (-1,%d) got (%d,%d))"%(self.ndim,pars_2d.shape[0],pars_2d.shape[1]))
    res = nm.zeros(pars_2d.shape[0],dtype=nm.double)
    i=0
    for apars in pars_2d:
      pars_proxy=nm.PyArray_ContiguousFromAny(apars,nm.NPY_DOUBLE,1,1)
      res[i] = clik_compute(self.celf,<double*> nm.PyArray_DATA(pars_proxy),self.err)
      er=doError(self.err)
      if er:
        raise er
      i+=1
    return res
    
  def __dealloc__(self):
    if self.celf!=NULL:
      clik_cleanup(&(self.celf))
      
  def get_has_cl(self):
    cdef int has_cl[6]
    clik_get_has_cl(self.celf, has_cl,self.err)
    er=doError(self.err)
    if er:
      raise er
    hcl = ""
    for i in range(6):
      hcl+=("0","1")[has_cl[i]]
    return hcl
  property has_cl:
    def __get__(self):
      return self.get_has_cl()
  
  
  def get_lmax(self):
    cdef int lmax[6]
    clik_get_lmax(self.celf, lmax,self.err)
    er=doError(self.err)
    if er:
      raise er
    lm = ()
    for i in range(6):
      lm+=(lmax[i],)
    return lm
    
  property lmax:
    def __get__(self):
      return self.get_lmax()
  
  def get_extra_parameter_names(self):
    cdef parname *names
    
    n_names = clik_get_extra_parameter_names(self.celf, &names, self.err)    
    res = ["%s"%names[i] for i in range(n_names)]
    stdlib.free(names)
    return tuple(res)

  def get_extra_parameter_names_by_lkl(self,int ilkl):
    cdef parname *names

    n_names = clik_get_extra_parameter_names_by_lkl(self.celf,ilkl, &names, self.err)    
    res = ["%s"%names[i] for i in range(n_names)]
    stdlib.free(names)
    return tuple(res)
    
  property extra_parameter_names:
    def __get__(self):
      return self.get_extra_parameter_names()

cdef extern:
  double*   c_camspec_get_fg "camspec_get_fg" (void* camclik,double *par,int lmax,error **err) 

def camspec_get_fg(nuis,lmax=3000):
  cdef double *res
  cdef error **err,*_err

  _err = NULL
  err = &_err

  pars_proxy=nm.PyArray_ContiguousFromAny(nuis,nm.NPY_DOUBLE,1,1)
  res = c_camspec_get_fg(NULL,<double*> nm.PyArray_DATA(pars_proxy),3000,err)
  er=doError(err)
  if er:
    raise er
  nes = nm.zeros((4,lmax))
  for i from 0<=i<4:
    for j from 0<=j<lmax:
      nes[i,j] = res[j*4+i]
  stdlib.free(res)
  return nes




##cdef extern from "fowly.h":
##  ctypedef struct powly:
##    double *H
##    double *a_bar
##    double *cl_fid
##    int neff,ncl_fid,tot_mode
##  powly* init_powly(int nside, char * ordering,
##                    unsigned char *mask_T, unsigned char *mask_P, 
##                    double *mapT,double *mapQ, double *mapU,
##                    double *Ndiag, double* N,int reduced,
##                    long lmax,int nell, int *ell,
##                    double *Cl,int *has_cl,
##                    int neff, double *U, double *G,
##                    error **err)
##  void free_powly(void **elf)
## 
##
##cdef extern from "string.h":
##      void * memcpy(void * s1,  void * s2, long n)
##
##def powly_javel(masks, maps,noise,ell,cl,has_cl,U,G=None):
##  cdef unsigned char *mask_T,*mask_P
##  cdef double *map_T,*map_Q, *map_U
##  cdef double *N, *Ndiag,*_U,*_G
##  cdef nm.ndarray[nm.uint8_t] mask_T_py,mask_P_py
##  cdef nm.ndarray[nm.double_t] map_T_py,map_Q_py,map_U_py,noisevar,Cl_proxy,a_bar,cl_fid
##  cdef nm.ndarray[nm.double_t,ndim=2] U_proxy,G_proxy,H
##  cdef nm.ndarray[nm.int32_t] ell_py,has_cl_py
##  cdef powly *ply
##  cdef error *_err,**err
##  
##  if len(masks)==2:
##    mask_T_py = nm.array(masks[0],dtype=nm.uint8)
##    mask_T = <unsigned char*> mask_T_py.data
##
##    mask_P_py = nm.array(masks[1],dtype=nm.uint8)    
##    mask_P = <unsigned char*> mask_P_py.data
##  else:
##    mask_T_py = nm.array(masks,dtype=nm.uint8)
##    mask_T = <unsigned char*> mask_T_py.data
##    mask_P = mask_T
##  
##  if len(maps)==3:
##    map_T_py = nm.array(maps[0],dtype=nm.double)
##    map_T = <double*> map_T_py.data
##    map_Q_py = nm.array(maps[1],dtype=nm.double)
##    map_Q = <double*> map_Q_py.data
##    map_U_py = nm.array(maps[2],dtype=nm.double)
##    map_U = <double*> map_U_py.data
##  else:
##    map_T_py = nm.array(maps,dtype=nm.double)
##    map_T = <double*> map_T_py.data
##    map_Q = NULL
##    map_U = NULL
##    
##  nside = int(nm.sqrt(nm.size(mask_T_py)/12))
##  
##  if isinstance(noise,(int,float)):
##    noisevar = nm.ones(12*nside**2*3,dtype=nm.double)*noise
##    Ndiag = <double*> noisevar.data
##    N = NULL
##    reduced = 0
##  else:
##    noisevar = nm.array(noise)
##    if noisevar.ndim==2:
##      N = <double*>noisevar.data
##      Ndiag = NULL
##    else:
##      Ndiag = <double*>noisevar.data
##      N = NULL
##
##  has_cl_py = nm.array([int(v) for v in has_cl],dtype=nm.int32)
##  Cl_proxy = nm.array(cl).flat[:]
##  lmax = len(Cl_proxy)/6-1
##  print lmax
##  if ell!=None:
##    #print ell
##    ell_py = nm.array(ell,dtype=nm.int32)
##  else:
##    ell_py = nm.arange(lmax+1,dtype=nm.int32)
##  nls = len(ell_py)
##  
##  
##  U_proxy = nm.array(U)
##  _U = <double*> U_proxy.data
##
##  _G = NULL
##  if G!=None:
##    G_proxy = nm.array(G)
##    _G = <double*> G_proxy.data
##  
##  neff = U_proxy.shape[1]
##    
##  _err = NULL
##  err = &_err
##  
##  ply = init_powly(nside,"ring",
##                   mask_T,mask_P,
##                   map_T,map_Q,map_U,
##                   Ndiag,N,0,
##                   lmax,nls,<int*>ell_py.data,<double*>Cl_proxy.data,<int*>has_cl_py.data,
##                   neff, _U, _G,err)
##    
##  er=doError(err)
##  if er:
##    raise er
##
##  a_bar = nm.zeros((ply.neff,),dtype=nm.double)
##  memcpy(a_bar.data,ply.a_bar,sizeof(double)*ply.neff)
##  H = nm.zeros((ply.tot_mode,ply.neff),dtype=nm.double)
##  memcpy(H.data,ply.H,sizeof(double)*ply.neff*ply.tot_mode)
##  cl_fid = nm.zeros((ply.ncl_fid,),dtype=nm.double)
##  memcpy(cl_fid.data,ply.cl_fid,sizeof(double)*ply.ncl_fid)
##  
##  free_powly(<void**>&ply)
##  
##  return cl_fid,a_bar,H
##  
##    
##  
##
##  