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

cdef extern from "clik_egfs.h":
  ctypedef char egfs_parstr[256]
  ctypedef char parname[256]
  ctypedef struct c_egfs "egfs":
    int nfr,nell
  c_egfs *egfs_init(int nvar, char **keyvars, int ndefaults, char** keys, char** values, int lmin, int lmax, double* cib_clustering,double *patchy_ksz, double *homogenous_ksz,double *tsz, double* cib_decor_clust, double * cib_decor_poisson,error **err)
  void egfs_compute(c_egfs *self, double *pars, double *rq, double *drq, error **err)
  void egfs_free(void **pelf)

cdef class egfs:
  cdef c_egfs* celf
  cdef int ndim,nfr,nell
  cdef object ids
  
  def __init__(self,parnames,pardef,lmin,lmax,freqs,inorm,cib_clustering=None,patchy_ksz=None,homogenous_ksz=None,tsz=None,cib_decor_clust=None,cib_decor_poisson=None,flter=None):
    cdef error *_err,**err
    cdef char *keys[100], *values[100], *keyvars[50]
    cdef double *_cib_clustering,*_patchy_ksz,*_homogenous_ksz,*_tsz,*_cib_decor_poisson,*_cib_decor_clust
    
    self.celf=NULL
    models = ["cib_clustering","cib_poisson","radio_poisson","tsz","ksz"]
    mardef = dict(pardef)
    frqstr = " ".join(["%s"%v for v in freqs])
    mardef.update(dict(zip(["eff_fr_"+v for v in models],[frqstr]*5)))
    mardef.update(dict(zip(["norm_fr_"+v for v in models],["%s"%"143"]*5)))
    mardef.update({"nfr":"%d"%len(freqs)})
    mkeys = mardef.keys()
    for i from 0<=i<len(mardef):
      keys[i] = mkeys[i]
      values[i] = mardef[mkeys[i]]
      
    self.celf=NULL
    _err = NULL
    err = &_err
    
    _cib_clustering = NULL
    _patchy_ksz = NULL
    _homogenous_ksz = NULL
    _tsz = NULL
    if cib_clustering!=None:
      cib_clustering_proxy=nm.PyArray_ContiguousFromAny(cib_clustering,nm.NPY_DOUBLE,1,1)
      patchy_ksz_proxy=nm.PyArray_ContiguousFromAny(patchy_ksz,nm.NPY_DOUBLE,1,1)
      homogenous_ksz_proxy=nm.PyArray_ContiguousFromAny(homogenous_ksz,nm.NPY_DOUBLE,1,1)
      tsz_proxy=nm.PyArray_ContiguousFromAny(tsz,nm.NPY_DOUBLE,1,1)
      _cib_clustering = <double*> nm.PyArray_DATA(cib_clustering_proxy)
      _patchy_ksz = <double*> nm.PyArray_DATA(patchy_ksz_proxy)
      _homogenous_ksz = <double*> nm.PyArray_DATA(homogenous_ksz_proxy)
      _tsz = <double*> nm.PyArray_DATA(tsz_proxy)
      
    _cib_decor_clust = NULL
    if cib_decor_clust!=None:
      cib_decor_clust_proxy=nm.PyArray_ContiguousFromAny(cib_decor_clust,nm.NPY_DOUBLE,2,2)
      assert cib_decor_clust_proxy.shape[0]==len(freqs)      
      assert cib_decor_clust_proxy.shape[1]==len(freqs)
      _cib_decor_clust = <double*> nm.PyArray_DATA(cib_decor_clust_proxy)
    _cib_decor_poisson = NULL
    if cib_decor_poisson!=None:
      cib_decor_poisson_proxy=nm.PyArray_ContiguousFromAny(cib_decor_poisson,nm.NPY_DOUBLE,2,2)
      assert cib_decor_poisson_proxy.shape[0]==len(freqs)      
      assert cib_decor_poisson_proxy.shape[1]==len(freqs)
      _cib_decor_poisson = <double*> nm.PyArray_DATA(cib_decor_poisson_proxy)
      
    if flter:
      farnames = [p for p,v in zip(parnames,flter) if v!=None]
      self.ids = [i for i,v in enumerate(flter) if v!=None]
    else:
      farnames = parnames
      self.ids = None

    for i from 0<=i<len(farnames):
      keyvars[i] = farnames[i]
    
    self.celf = egfs_init(len(farnames),keyvars,len(mardef),keys,values,lmin,lmax,_cib_clustering,_patchy_ksz,_homogenous_ksz,_tsz,_cib_decor_clust,_cib_decor_poisson,err)
    
    er=doError(err)
    if er:
      raise er
    
    self.ndim = len(parnames)
    self.nfr = self.celf.nfr
    self.nell = self.celf.nell

  def __call__(self,pars):
    cdef error *_err,**err
    cdef double *_drq,*_rq
    if len(pars)!=self.ndim:
      raise Exception("Bad shape (expecting (%d) got (%d))"%(self.ndim,len(pars)))
    rq = nm.zeros((self.nfr,self.nfr,self.nell),dtype=nm.double)
    sars = pars
    drqa = nm.zeros((self.nfr,self.nfr,self.nell,self.ndim),dtype=nm.double)
    drq = drqa
    if self.ids!=None:
      drqb = nm.zeros((self.nfr,self.nfr,self.nell,len(self.ids)),dtype=nm.double)
      drq = drqb
      sars  =nm.array([pars[i] for i in self.ids])
    pars_proxy=nm.PyArray_ContiguousFromAny(sars,nm.NPY_DOUBLE,1,1)
    _err = NULL
    err = &_err
    egfs_compute(self.celf,  <double*> nm.PyArray_DATA(pars_proxy), <double*> nm.PyArray_DATA(rq),<double*> nm.PyArray_DATA(drq), err);
    er=doError(err)
    if er:
      raise er
    if self.ids!=None:
      drqa[:,:,:,self.ids] = drqb
    return rq,drqa
    
  def __dealloc__(self):
    if self.celf!=NULL:
      egfs_free(<void**>&(self.celf))
    
def default_models(defmodels=[],varmodels=[],varpars=[],defvalues={},dnofail=True,reduce=False):
  prs = """
  #-> cib clustering
  alpha_dg_cl     = 3.8
  tilt_dg_cl      = 0.8
  norm_dg_cl      = 6
  
  #-> cib poisson  
  alpha_dg_po     = 3.8
  sigma_dg_po     = 0.4
  norm_dg_po      = 9
  fpol_dg_po      = 0.01
  
  #-> radio poisson
  # Updated default values for best match with Paoletti et al. (De
  # Zotti et al.) model. Commented values are original Millea et
  # al. values.

  # alpha_rg        = -0.5
  # sigma_rg        = 0.1
  # norm_rg         = 133
  alpha_rg        = -0.36
  sigma_rg        = 0.64
  norm_rg         = 78.5
  gamma_rg        = -0.8
  fpol_rg         = 0.05
  
  #-> tsz
  tsz_pca1        = 0
  tsz_pca2        = 0
  tsz_mean_scale  = 1
  
  #-> ksz
  norm_ov         = 1
  norm_patchy     = 1
  shift_patchy    = 1
  """  
  lprs = prs.split("\n")
  
  iii = [i for i,vv in enumerate(lprs) if vv.strip() and vv.strip()[:3]=="#->"]
  pfs = {}
  for i,j in zip(iii,iii[1:]+[-1]):
    nmod = lprs[i].replace("#->","").strip()
    pn = [vv.split("=")[0].strip() for vv in lprs[i:j] if vv.strip() and vv.strip()[0]!="#"]  
    pfs[nmod]=pn

  pv = [float(vv.split("=")[1].strip()) for vv in lprs if vv.strip() and vv.strip()[0]!="#"]  
  pn = [vv.split("=")[0].strip() for vv in lprs if vv.strip() and vv.strip()[0]!="#"]  

  aps = dict(zip(pn,pv))
  if varpars==[]:
    if varmodels==[]:
      varmodels = pfs.keys()
    varpars = []
    for mm in varmodels:
      varpars = varpars + pfs[mm]
  
  varmodels = set(varmodels)
  for vp in varpars:
    for kk in pfs.keys():
      if vp in pfs[kk]:
        varmodels.add(kk)
    
  dmm = set(defmodels)  
  dmm.update(varmodels)
  for vp in defvalues.keys():
    for kk in pfs.keys():
      if vp in pfs[kk]:
        dmm.add(kk)
    
  defs = {}
  for dm in dmm:
    for pn in pfs[dm]:
      if pn not in varpars:
        defs[pn] = str(aps[pn])
  defs.update(defvalues)
  if dnofail or reduce:
    rv = [aps.get(pn,None) for pn in varpars]
  else:
    rv = [aps[pn] for pn in varpars]
  if reduce:
    varpars = [pn for pn,pv in zip(varpars,rv) if pv!=None]
    rv = [pv for pv in rv if pv!=None]

  return defs,varpars,rv
  
def init_defaults(datapath,defmodels=[],varmodels=[],varpars=[],defvalues={},dnofail=False,reduce=True):
  import os.path as osp
  
  defs = {}
  defs["template_cib_clustering"]=osp.join(datapath,"clustered_1108.4614.dat")
  defs["template_patchy_ksz"]=osp.join(datapath,"ksz_patchy.dat")
  defs["template_homogenous_ksz"]=osp.join(datapath,"ksz_ov.dat")
  defs["template_tsz"]=osp.join(datapath,"tsz.dat")

  defs["rg_flux_cut"]="330"
  defs["norm_rg_flux_cut"]="330"
  
  oefs,varpars,pv = default_models(defmodels,varmodels,varpars,defvalues,dnofail,reduce)
  
  defs.update(oefs)
    
  return defs,varpars,pv
    
def simple_egfs(lmin,lmax,freq,norm_freq,varpars=[],varmodels=[],defmodels=[],datapath="./",defs={}):
  oefs,pn,pv =  init_defaults(datapath,defmodels,varmodels,varpars)
  oefs.update(defs)
  megfs = egfs(pn,oefs,lmin,lmax,freq,freq.index(norm_freq))
  return megfs,pv
  
def testme(datapath="./"):
  megfs,pv = simple_egfs(0,4000,[100,143,217,353,545,857],143,datapath=datapath)
  
  return megfs,pv

try:
  def add_xxx(agrp,vpars,defaults,values,lmin,lmax,template_names,tpls,cib_decor_clustering):     
    import parobject as php
    agrp.attrs["ndim"] = len(vpars)
    agrp.attrs["keys"] = php.pack256(*vpars)
    
    agrp.attrs["ndef"] = len(defaults)
    agrp.attrs["defaults"] = php.pack256(*defaults)
    agrp.attrs["values"] = php.pack256(*values)

    agrp.attrs["lmin"] = lmin
    agrp.attrs["lmax"] = lmax

    for nnm,vvv in zip(template_names,tpls):
      agrp.create_dataset(nnm, data=vvv.flat[:])

    if cib_decor_clustering!=None:
      agrp.attrs["cib_decor_clustering"] = cib_decor_clustering.flat[:]

    return agrp  
except Exception:
  pass
    
def build_decor_step(frqs,step):
  ll = len(frqs)
  import itertools as itt
  chl = [0]+[ii+1 for ii,l1,l2 in zip(itt.count(),frqs,frqs[1:]) if l1!=l2]+[len(frqs)]
  #make block
  import numpy as nm
  mat = nm.zeros((ll,ll))
  for ipp0 in range(len(chl)-1):
    p0,p1=chl[ipp0],chl[ipp0+1]
    for ipp2 in range(len(chl)-1):
      p2,p3=chl[ipp2],chl[ipp2+1]
      away = ipp0-ipp2
      mat[p0:p1,p2:p3] = step**away
      mat[p2:p3,p0:p1] = step**away
  return mat
