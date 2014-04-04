#! PYTHONEXE
import sys
sys.path = ["REPLACEPATH"]+sys.path

import numpy as nm
import clik
import clik.hpy as hpy
import clik.parobject as php


def main(argv):
  if len(sys.argv)<4:
    print "usage : %s lkl_file_1 lkl_file_2 [lkl_file_3 ...] result_lkl_file"
    sys.exit(1)

  for fl in sys.argv[1:-1]:
    if clik.try_lensing(fl):
      print "clik_join doesn't work yet with lensing likelihood, sorry"
      sys.exit(1)
  
  
  lkls = [hpy.File(ll)["clik"] for ll in sys.argv[1:-1]]
  reslkl = hpy.File(sys.argv[-1],"w",ty=lkls[0])
  
  nlkl = 0
  lmax = -nm.ones(6,dtype=nm.int)
  resclik = reslkl.create_group("clik")
  name = []
  loc = []
  defn = []
  defloc = []
  var = nm.zeros((0,0))

  for lklin in lkls:
    if "prior" in lklin:
      prid = lklin["prior"]
      pname = [n.strip() for n in prid.attrs["name"].split('\0') if n]
      for n in name:
        if n in pname:
          raise Exception("already got a prior on %s"%n)
      ploc = prid["loc"][:]
      pvar = prid["var"][:]
      if len(pvar)==len(ploc):
        pvar = nm.diag(pvar)
      pvar.shape = (len(ploc),-1)
      nvar = nm.zeros((len(loc)+len(ploc),len(loc)+len(ploc)))
      nvar[:len(loc),:len(loc)] = var
      nvar[len(loc):,len(loc):] = pvar
      var = nvar
      name = list(name) + list(pname)
      loc = nm.concatenate((loc,ploc))


    lmaxin = lklin.attrs["lmax"]
    lmax = nm.max((lmax,lmaxin),0)
    nlklin = lklin.attrs["n_lkl_object"]
    
    for i in range(nlklin):
      grpin = "lkl_%d"%i
      grpout = "lkl_%d"%nlkl
      nlkl+=1
      lklin.copy(lklin[grpin],resclik,grpout)
    
    if "default" in lklin:
      prid = lklin["default"]
      pname = [n.strip() for n in prid.attrs["name"].split('\0') if n]
      ploc = prid["loc"][:]
      for i,n in enumerate(pname):
        l = ploc[i]
        add=True
        for ii,nn in enumerate(defn):
          if n==nn:
            if l!=defloc[ii]:
              raise Exception("cannot fix the same parameter with different values")
            add=False
            break
        if add:
          defn+=[n]
          defloc +=[l]

    
  if len(name):
    prid = resclik.create_group("prior")
    prid.attrs["name"] = php.pack256(*name)
    prid.create_dataset("loc", data=loc.flat[:])
    prid.create_dataset("var", data=var.flat[:])
  if len(defn):
    prid = resclik.create_group("default")
    prid.attrs["name"] = php.pack256(*defn)
    prid.create_dataset("loc", data=nm.array(defloc).flat[:])
    
  resclik.attrs["lmax"] = lmax.astype(nm.int)
  resclik.attrs["n_lkl_object"] = nlkl
  
  reslkl.close()
    
if __name__=="__main__":
  main(sys.argv)