import parobject as php
import numpy as nm


def base_smica(root_grp,hascl,lmin,lmax,nT,nP,wq,rqhat,Acmb,rq0=None,bins=None):
  if bins==None:
    nbins = 0
  else:
    bins.shape=(-1,(lmax+1-lmin)*nm.sum(hascl))
    nbins = bins.shape[0]
  lkl_grp = php.add_lkl_generic(root_grp,"smica",1,hascl,lmax,lmin,nbins = nbins,bins = bins.flat[:])

  lkl_grp.attrs["m_channel_T"] = nT
  lkl_grp.attrs["m_channel_P"] = nP
  lkl_grp.create_dataset('wq', data=wq)
  lkl_grp.create_dataset("Rq_hat",data=rqhat.flat[:])
     
  if rq0 !=None:
    lkl_grp.create_dataset("Rq_0",data=rq0.flat[:])
  
  lkl_grp.attrs["A_cmb"] = Acmb
  lkl_grp.attrs["n_component"] = 1  
  
  return lkl_grp

def add_component(lkl_grp,typ,position=-1):
  nc = lkl_grp.attrs["n_component"]
  if position ==-1:
    position = nc
  assert position <=nc
  for ic in range(nc,position,-1):
    lkl_grp.copy("component_%d"%(ic-1),"component_%d"%ic)
    del lkl_grp["component_%d"%(ic-1)]
  agrp = lkl_grp.create_group("component_%d"%(position))  
  agrp.attrs["component_type"]=typ
  lkl_grp.attrs["n_component"] = nc+1
  return agrp

def add_cst_component(lkl_grp,rq0,position=-1):
  agrp = add_component(lkl_grp,"cst",position)
  agrp.create_dataset("Rq_0",data=rq0.flat[:])
  return agrp
def add_cst_component_pars(lkl_grp,pars):
  rq0 = php.read_somearray(pars.rq0)
  return add_cst_component(lkl_grp,rq0)

def add_gcal_component(lkl_grp,typ,ngcal,gcaltpl,binned=False,names=[],position=-1):
  if typ.lower() == "log":
    typ = "gcal_log"
  else:
    typ = "gcal_lin"
  agrp = add_component(lkl_grp,typ,position)
  agrp.attrs["ngcal"] = nm.array(ngcal,dtype=nm.int) 
  agrp.create_dataset("gcaltpl",data=nm.array(gcaltpl,dtype=nm.double).flat[:])
  if binned:
    agrp.attrs["binned"]=1
  else:
    agrp.attrs["binned"]=0

  if names:
    setnames(agrp,names)
  return agrp

def read_gcal_data(pars,lkl_grp):
  return read_gcal_data_(pars.str_array.datacal,pars.float_array.ngcal,pars.int_array(default=[-1]).lmax_tpl,lkl_grp)

def read_gcal_data_(datacal,ngcal,lmax_tpl,lkl_grp):
  # returns the dtemplate data
  lmin = lkl_grp.attrs["lmin"]
  lmax = lkl_grp.attrs["lmax"]
  if len(datacal) == 1:
    dat = nm.loadtxt(datacal[0]).flat[:]
  else:
    assert len(datacal)==len(ngcal)
    dat = ()
    i=0
    if len(lmax_tpl)==1:
      lmax_tpl = list(lmax_tpl)*len(ngcal)
    assert len(ngcal)==len(lmax_tpl)
    for ff,lm in zip(datacal,lmax_tpl):
      assert lm>=lmax
      idat = nm.loadtxt(ff).flat[:]
      if lm!=-1:
        idat.shape = (-1,lm+1)
        idat = idat[:ngcal[i],lmin:lmax+1]
        idat = idat.flat[:]
      dat = nm.concatenate((dat,idat))  
  return dat

def add_gcal_component_pars(lkl_grp,pars):
  typ = pars.str.type
  ngcal = pars.float_array.ngcal  
  gcaltpl = read_gcal_data(pars,lkl_grp)
  names = []
  if "name" in pars:
    names = pars.str_array.name
    assert len(names) == nm.sum(ngcal)
  binned = bool(pars.int(default=0).binned!=0)
  return add_gcal_component(lkl_grp,typ,ngcal,galtpl,binned,names)

def setnames(agrp,names):
  agrp.attrs["names"] = php.pack256(*names) 
  
def add_egfs_component(lkl_grp,vpars,defaults,values,lmin,lmax,template_names,tpls,cib_decor_clustering,position=-1):
  import egfs
  agrp = add_component(lkl_grp,"egfs",position)
  egfs.add_xxx(agrp,vpars,defaults,values,lmin,lmax,template_names,tpls,cib_decor_clustering)
  agrp.attrs["A_cmb"] = lkl_grp.attrs["A_cmb"]
  return agrp

def add_from_pars(lkl_grp,parfile):
  import miniparse
  pars = miniparse.miniparse(parfile)
  typ = pars.str.ctype
  return globals()["add_%s_component_pars"](lkl_grp,pars)

def add_parametric_component(lkl_grp,name,dets,vpars,lmin,lmax,defaults={},color=None,rename={},voidmask="",data=None,position=-1):
  import os
  import parametric
  import os.path as osp
  parametric.register_all(parametric.__dict__)

  # initialize parameters
  pm = getattr(parametric,name)(dets,vpars,lmin,lmax,defaults,color=color,rename=rename,voidmask=voidmask)
  #filter them out
  npars = [vp for vp in vpars if pm.has_parameter(vp)]
  agrp = add_component(lkl_grp,name,position)
  agrp.attrs["ndim"] = len(npars)
  agrp.attrs["keys"] = php.pack256(*npars)
  
  agrp.attrs["ndef"] = len(defaults)
  defkey = defaults.keys()
  defval = [defaults[k] for k in defkey]
  agrp.attrs["defaults"] = php.pack256(*defkey)
  agrp.attrs["values"] = php.pack256(*defval)

  agrp.attrs["lmin"] = lmin
  agrp.attrs["lmax"] = lmax

  agrp.attrs["dfreq"] = [float(d) for d in dets]
  agrp.attrs["A_cmb"] = lkl_grp.attrs["A_cmb"]

  if voidmask:
      _voidlist = [i for i in range(len(voidmask)) if not bool(int(voidmask[i]))]
      nvoid = len(_voidlist)
      if nvoid!=0:
        agrp.attrs["nvoid"] = nvoid
        agrp.attrs["voidlist"] = _voidlist
      
  if color is not None:
    agrp.create_dataset("color",data=color.flat[:])  

  template = pm.get_template()
  if template is None:
    pass
  else:
    if data is None:
      agrp.create_dataset("template",data=nm.array(template,dtype=nm.double).flat[:])  
    else:
      agrp.create_dataset("template",data=nm.array(data,dtype=nm.double).flat[:])

  if rename:
    rename_from = rename.keys()
    rename_to = [rename[k] for k in rename_from]
    agrp.attrs["rename_from"] = php.pack256(*rename_from)
    agrp.attrs["rename_to"] = php.pack256(*rename_to)
    agrp.attrs["nrename"] = len(rename_from)
  return agrp

import numpy as nm

def set_criterion(lkl_grp,typ,**extra):
  print typ
  if typ.lower()=="classic":
    lkl_grp.attrs["criterion"]="classic"
    return
  if typ.lower()=="eig":
    lkl_grp.attrs["criterion"]="eig"
    if "eig_norm" in extra:
      lkl_grp["criterion_eig_norm"]=extra["eig_nrm"]
    else:
      import numpy.linalg as la
      import numpy as nm
      rqh = lkl_grp["Rq_hat"][:]
      nq = len(lkl_grp["wq"][:])
      m = lkl_grp.attrs["m_channel_T"] + lkl_grp.attrs["m_channel_P"] 
      rqh.shape=(nq,m,m)
      nrm = nm.array([.5*(nm.log(nm.abs(la.det(rqh[i])))+m) for i in range(nq)])
      lkl_grp["criterion_eig_norm"] = nrm
    return 
  if typ.lower()=="gauss":
    import numpy.linalg as la
    import numpy as nm
    rqh = lkl_grp["Rq_hat"][:]
    nq = len(lkl_grp["wq"][:])
    m = lkl_grp.attrs["m_channel_T"] + lkl_grp.attrs["m_channel_P"] 
    rqh.shape=(nq,m,m)
    nrm = nm.array([.5*(nm.log(nm.abs(la.det(rqh[i])))+m) for i in range(nq)])
    lkl_grp["criterion_eig_norm"] = nrm
    return 
  if typ.lower()=="quad":
    import numpy as nm
    if "fid" in extra:
      if "mask" in extra:
        lkl_grp["criterion_quad_mask"] = extra["mask"].flat[:]
        mask = extra["mask"].flat[:]
        rq = extra["fid"]*1.
        n = int(nm.sqrt(mask.size))
        rq.shape=(-1,n,n)
        mask.shape=(n,n)
        sn = int(nm.sum(nm.triu(mask)))
        fq = nm.zeros((len(rq),sn,sn))
        for i in range(len(rq)):
          ifq = build_tensormat(rq[i],mask)
          fq[i] = nm.linalg.inv(ifq)
        lkl_grp["criterion_quad_mat"] = fq.flat[:]
      else:
        lkl_grp["criterion_quad_mat"]=extra["fid"].flat[:]
    lkl_grp.attrs["criterion"]="quad"

  return
  
def build_tensormat(rq ,mask=None):
  n = len(rq)
  if mask==None:
    mask = nm.ones((n,n))
  M = nm.zeros((n**2,n**2))
  for i in range(n):
    for j in range(n):
      for k in range(n):
        for l in range(n):
          M[j*n+k,l*n+i] = rq[i,j]*rq[k,l]
  B = build_vecproj(mask)
  return nm.dot(B.T,nm.dot(M,B))

def build_vecproj(mask):
  n=len(mask)
  p=0
  B = nm.zeros((n**2,nm.sum(nm.triu(mask))))
  for i in range(n):
    for j in range(i,n):
      if mask[i,j]==0:
        continue
      B[i*n+j,p]=.5
      B[j*n+i,p]=.5
      if i==j:
        B[i*n+j,p]=1
      p+=1
  return B


def parametric_from_smica(h5file,lmin=-1,lmax=-1,ilkl=0):
  import hpy
  ff = hpy.File(h5file,"r")
  return parametric_from_smica_group(ff["clik/lkl_%d"%ilkl],lmin,lmax)
  ff.close()

def parametric_from_smica_group(hgrp,lmin=-1,lmax=-1):
  import parametric as prm
  nc = hgrp.attrs["n_component"]
  prms = []
  for i in range(1,nc):
    compot = hgrp["component_%d"%i].attrs["component_type"]
    if not (compot in dir(prm)):
      continue
    key = [v.strip() for v in hgrp["component_%d"%i].attrs["keys"].split("\0") if v.strip() ]
    default = [v.strip() for v in hgrp["component_%d"%i].attrs["defaults"].split("\0") if v.strip() ]
    value = [v.strip() for v in hgrp["component_%d"%i].attrs["values"].split("\0") if v.strip() ]
    defdir = dict(zip(default,value))
    frq = hgrp["component_%d"%i].attrs["dfreq"]
    try:
      rename_from = [v.strip() for v in hgrp["component_%d"%i].attrs["rename_from"].split("\0") if v.strip() ]
      rename_to = [v.strip() for v in hgrp["component_%d"%i].attrs["rename_to"].split("\0") if v.strip() ]
      rename = dict(zip(rename_from,rename_to))
    except Exception,e:
      rename = {}
    #print rename
    #print key
    #print default
    #print value
    try:
      color = hgrp["component_%d/color"%i][:]
    except Exception,e:
      color = None
    try:
      data = hgrp["component_%d/data"%i][:]
    except Exception,e:
      data = None
    if lmin==-1:
      lmin = hgrp["component_%d"%i].attrs["lmin"]
    if lmax==-1:
      lmax = hgrp["component_%d"%i].attrs["lmax"]
    if data == None:
      prms += [getattr(prm,compot)(frq,key,lmin,lmax,defdir,rename=rename,color=color)]
    else:
      prms += [getattr(prm,compot)(frq,key,lmin,lmax,defdir,rename=rename,color=color,data=data)]
  return prms  