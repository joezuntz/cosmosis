#! PYTHONEXE
import sys
sys.path = ["REPLACEPATH"]+sys.path

import numpy as nm
import clik.parobject as php
import clik
import re
import os.path as osp

def main(argv):
  pars = clik.miniparse(argv[1])
  try:
    test_cl = nm.loadtxt(osp.join(pars.wmap_data,"data/test_cls_v4.dat"))
  except IOError,e:
    test_cl = nm.loadtxt(osp.join(pars.wmap_data,"data/test_cls_v5.dat"))

  mcl = nm.zeros((4,1201),dtype=nm.double)
  llp1s2pi = nm.arange(1201)*nm.arange(1,1202)/2./nm.pi
  mcl[:,2:] = (test_cl[:1201-2,1:5].T)/llp1s2pi[2:]
  
  ttmin = max(2,pars.int.ttmin)
  ttmax = min(1200,pars.int.ttmax)
  temin = max(2,pars.int.temin)
  temax = min(800,pars.int.temax)
  
  has_tt = True
  has_te = True
  
  if pars.int.ttmin>pars.int.ttmax:
    ttmin = 1201
    ttmax = 2
    has_tt = False
  if pars.int.temin>pars.int.temax:
    temin = 801
    temax = 2
    has_te = False
  
  #print has_tt,has_te,ttmin,ttmax,temin,temax
  
  root_grp,hf = php.baseCreateParobject(pars.res_object)
  hascl = [0]*6
  hascl[0] = has_tt
  hascl[1:4] = [has_te]*3
  hascl = nm.array(hascl,dtype=nm.int)
  #print hascl
  
  lmin = 0
  lmax = min(1200,max(pars.int.ttmax,pars.int.temax))
  
  mcl = (nm.compress(hascl[:4],mcl[:,:lmax+1],0)).flat[:]
  
  lkl_grp = php.add_lkl_generic(root_grp,"wmap",1,hascl,lmax,lmin)
  
  lkl_grp.attrs["ttmin"] = ttmin
  lkl_grp.attrs["temin"] = temin
  lkl_grp.attrs["ttmax"] = ttmax
  lkl_grp.attrs["temax"] = temax
  
  lkl_grp.attrs["use_gibbs"] = pars.int.use_gibbs
  lkl_grp.attrs["use_lowl_pol"] = pars.int.use_lowl_pol
  
  #lkl_grp.attrs["external_dir"] = osp.realpath(pars.wmap_data)
  php.add_external_data(osp.realpath(pars.wmap_data),lkl_grp,tar=bool(pars.int(default=1).include))

  hf.close()
  
  if hasattr(clik,"clik"):
    res = php.add_selfcheck(pars.res_object,mcl)
    print "lkl for init cl %g"%res
  
  if "cl_save" in pars:
    f=open(pars.cl_save,"w")
    for ci in mcl:
      print >>f,ci
    f.close()

import sys
if __name__=="__main__":
  main(sys.argv)