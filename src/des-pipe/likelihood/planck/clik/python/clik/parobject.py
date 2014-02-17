import numpy as nm
import hpy
import shutil

def pack256(*li):
  rr=""
  for l in li:
    rr += l+'\0'*(256-len(l))
  return rr

def baseCreateParobject(parobject):
  # init file
  hf = hpy.File(parobject, 'w')
  root_grp = hf.create_group("clik")
  
  # fill general info
  root_grp.attrs["n_lkl_object"] = 0
  root_grp.attrs["lmax"] = [-1,-1,-1,-1,-1,-1]
  
  return root_grp,hf
  
def add_external_data(directory,lkl_grp,tar=False):
  import os.path as osp
  import os
  import tempfile
  import tarfile
  import numpy as nm
  if not tar:
    lkl_grp.attrs["external_dir"] = osp.realpath(directory)
  else:
    if hpy.is_h5py_object(lkl_grp):
      tmp = tempfile.TemporaryFile()
      tartmp = tarfile.TarFile(mode = "w", fileobj=tmp)
      cd = os.getcwd()
      os.chdir(directory)
      for d in os.listdir("."):
        if d not in [".",".."]:
          tartmp.add(d) 
      tartmp.close()
      tmp.seek(0)
      dat = nm.frombuffer(tmp.read(),dtype=nm.uint8)
      lkl_grp.create_dataset("external_data",data=dat.flat[:])
      tmp.close()
      os.chdir(cd)
    else:
      #os.mkdir(lkl_grp._name+"/_external")
      shutil.copytree(directory,lkl_grp._name+"/_external")    
      lkl_grp.attrs["external_dir"] = "."
      
def add_lkl_generic(root_grp,lkl_type,unit,has_cl,lmax=-1,lmin=-1,ell=None,wl=None,nbins=0,bins=None,compress_bns=True):
  ilkl = root_grp.attrs["n_lkl_object"]
  lmaxs = root_grp.attrs["lmax"]
  name = "lkl_%d"%ilkl
  
  lkl_grp = root_grp.create_group(name)
  lkl_grp.attrs["lkl_type"] = lkl_type
  lkl_grp.attrs["unit"]     = unit
  lkl_grp.attrs["has_cl"]   = has_cl
  ncl = nm.sum(has_cl)
  
  assert not (lmax == -1 and ell == None)
  assert not (lmax != -1 and ell != None)

  if ell != None:
    lkl_grp.attrs["ell"] = nm.sort(ell)
    lmax = max(ell)
  
  if lmax>-1:
    lkl_grp.attrs["lmax"] = lmax
    ell = nm.arange(lmax+1)
    if lmin>-1:
      lkl_grp.attrs["lmin"] = lmin
      ell = ell[lmin:]
      
  
  
  if nbins>0:
    lkl_grp.attrs["nbins"] = int(nbins)
    if compress_bns==True:
      ish = bins.shape
      bins.shape=(nbins,-1)

      try:
        b_ws,blmin,blmax = compress_bins(bins)
        bins.shape=ish
        if b_ws.size+2*blmin.size<bins.size:
          print "compressing bins"
          lkl_grp.create_dataset("bin_ws",data=b_ws.flat[:])
          lkl_grp.create_dataset("bin_lmin",data=blmin.flat[:])
          lkl_grp.create_dataset("bin_lmax",data=blmax.flat[:])
        else:
          compress_bns=False
      except Exception:
        compress_bns=False 
    if compress_bns==False:
      lkl_grp.create_dataset("bins",data=bins.flat[:])

  if wl!=None:
    lkl_grp.attrs["wl"] = wl
    
  ilkl+=1
  root_grp.attrs["n_lkl_object"] = ilkl
  lmaxs = [max(lm,((lmax+1)*hcl)-1) for lm,hcl in zip(lmaxs,has_cl)]
  root_grp.attrs["lmax"] = lmaxs
    
  return lkl_grp
  
def compress_bins(bins):
  mins = bins!=0
  l = nm.arange(bins.shape[-1])
  # there is an issue with TEB stuff here.
  #to be fixed later
  blmin,blmax = nm.array([(l[ins][0],l[ins][-1]) for ins in mins]).T
  b_ws = bins[mins]
  return b_ws,blmin,blmax
  
def uncompress_bins(shape,b_ws,blmin,blmax):
  bins = nm.zeros(shape)
  lc = 0
  for i in range(shape[0]):
    bsz = blmax[i]-blmin[i]+1
    bins[i,blmin[i]:blmax[i]+1] = b_ws[lc:lc+bsz]
    lc+=bsz
  return bins

def read_ell(lkl_grp):
  if ell in lkl_grp.attrs:
    return lkl_grp.attrs["ells"]
  else:
    lmax = lkl_grp.attrs["lmax"]
    ell = nm.arange(lmax+1)
    if "lmin" in lkl_grp.attrs:
      lmin = lkl_grp.attrs["lmin"]
      ell = ell[lmin:]
    return ell

def read_bins(lkl_grp):
  if "bins" in lkl_grp:
    bins = lkl_grp["bins"][:]
    bins.shape = (lkl_grp.attrs["nbins"],-1)
  else:
    ell = read_ell(lkl_grp)
    shape = (lkl_grp.attrs["nbins"],len(ell))
    return uncompress_bins(shape,lkl_grp["bin_ws"],lkl_grp["bin_lmin"],lkl_grp["bin_lmax"])

def add_selfcheck(fname,pars):
  import lkl
  mlkl = lkl.clik(fname)
  res = mlkl(pars)
  del(mlkl)
  
  # add check pars
  hf = hpy.File(fname, 'r+')
  root_grp = hf["clik"]
  root_grp.create_dataset("check_param",data=pars)
  root_grp.attrs["check_value"] = float(res)
  hf.close()
  return res

def remove_selfcheck(fname=None,root_grp=None):
  if fname!=None:
    hf = hpy.File(fname, 'r+')
    root_grp = hf["clik"]
  if "check_param" in root_grp:
    del root_grp["check_param"]
  if "check_value" in root_grp:
    del root_grp["check_value"]
  if fname:
    hf.close()
    
def read_somearray(somepath):
  # for now only ascii arrays
  try:
    import piolib as pio
    return pio.read(somepath)
  except Exception:
    return nm.loadtxt(somepath) 

def copy_and_get_0(pars):
  if "input_object" in pars:
    hpy.copyfile(pars.input_object,pars.res_object)
  outhf = hpy.File(pars.res_object,"r+")
  return outhf,outhf["clik/lkl_%d"%pars.int(default=0).lkl_id]

def add_pid(lkl_grp,pid=""):
  if not pid:
    import uuid
    pid = str(uuid.uuid4())
  lkl_grp.attrs["pipeid"]=pid

def add_prior(root_grp,name,loc,var):
  assert len(name)==len(loc)
  assert len(name)==len(var) or len(name)**2==len(var)
  pred = {}
  if "default" in root_grp:
    prid = root_grp["default"]
    pred = dict(zip([v.strip() for v in prid.attrs["name"].split("\0") if v.strip()],prid["loc"][:]))
    del(prid.attrs["name"])
    del[prid["loc"]]
  if len(var) == len(loc):
      var = nm.diag(var)
  if "prior" in root_grp:
    prid = root_grp["prior"]
    pname = [n.strip() for n in prid.attrs["name"].split()]
    for n in name:
      if n in pname:
        raise Exception("already got a prior on %s"%n)
    ploc = prid["loc"][:]
    pvar = prid["var"][:]
    if len(pvar)==len(ploc):
      pvar = nm.diag(pvar)
    pvar.shape = (len(ploc),-1)
    nvar = nm.zeros((len(loc),len(loc)))
    nvar[:len(loc),:len(loc)] = var
    nvar[len(loc):,len(loc):] = pvar
    var = nvar
    name = list(name) + list(pname)
    print name
    loc = nm.concatenate((loc,ploc))
  else:
    prid = root_grp.create_group("prior")
  
  var.shape = (len(loc),-1)
  if nm.alltrue(var==nm.diag(nm.diagonal(var))):
    var = nm.diagonal(var)
  prid.attrs["name"] = pack256(*name)
  prid.create_dataset("loc", data=loc.flat[:])
  prid.create_dataset("var", data=var.flat[:])
  if pred:
    nam = pred.keys()
    lo = [pred[k] for k in nam]
    add_default(root_grp,nam,lo)

def add_free_calib(root_grp,name):
  root_grp.attrs["free_calib"] = name
  
def add_default(root_grp,name,loc,extn=None):

  if "default" in root_grp:
    prid = root_grp["default"]
    #print prid.keys()
    #print prid.attrs.keys()
    pred = dict(zip([v.strip() for v in prid.attrs["name"].split("\0") if v.strip()],prid["loc"][:]))
    print pred
  else:
    prid = root_grp.create_group("default")
    pred = {}
  
  pred.update(dict(zip(name,loc)))

  if extn !=None:
    for n in name:
      if n not in extn and n not in pred:
        raise Exception("extra parameter %s does not exist"%(n))

  fname = pred.keys()
  floc = nm.array([pred[n] for n in fname])
  prid.attrs["name"] = pack256(*fname)
  if "loc" in prid.keys():
    del(prid["loc"])
  prid["loc"]=floc.flat[:]

  if "prior" in root_grp:
    prid = root_grp["prior"]
    pname = [n.strip() for n in prid.attrs["name"].split('\0') if n.strip()]
    ploc = prid["loc"][:]
    pvar = prid["var"][:]
    if len(pvar)==len(ploc):
      pvar = nm.diag(pvar)
    pvar.shape = (len(ploc),-1)
    print pname,fname
    idx = [i for i,n in enumerate(pname) if n not in fname]
    print idx
    if len(idx)!=len(ploc):
      ploc = ploc[idx] 
      pvar = pvar[idx][:,idx]
      pname = [pname[i] for i in idx]
      del(prid.attrs["name"])
      del(prid["loc"])
      del(prid["var"])
      if len(ploc):
        prid.attrs["name"] = pack256(*pname)
        prid.create_dataset("loc", data=ploc.flat[:])
        prid.create_dataset("var", data=pvar.flat[:])
      else:
        del(prid)
        del(root_grp["prior"])
