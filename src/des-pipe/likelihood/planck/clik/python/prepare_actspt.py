#! PYTHONEXE
import sys
sys.path = ["REPLACEPATH"]+sys.path

import numpy as nm
import numpy.random as ra
import numpy.linalg as la
import clik.parobject as php
import clik
import re
import os.path as osp

def main(argv):
  pars = clik.miniparse(argv[1])
  #test_cl = nm.loadtxt(osp.join(pars.wmap_data,"data/v2a1s_best_lcdm_6000.txt"))
  
  #mcl = nm.zeros((4,1201),dtype=nm.double)
  #llp1s2pi = nm.arange(1201)*nm.arange(1,1202)/2./nm.pi
  #mcl[:,2:] = (test_cl[:1201-2,1:].T)/llp1s2pi[2:]
  
  lmin11 = max(2,pars.int(default=1000).lmin11)
  lmin12 = max(2,pars.int(default=1500).lmin12)
  lmin22 = max(2,pars.int(default=1500).lmin22)
  lmax11 = min(10000,pars.int(default=10000).lmax11)
  lmax12 = min(10000,pars.int(default=10000).lmax12)
  lmax22 = min(10000,pars.int(default=10000).lmax22)
  
  tt_lmax_mc = min(10000,pars.int(default=5000).tt_lmax_mc)
  
  
  lmin = min(2,lmin11,lmin12,lmin22)
  lmax = min(tt_lmax_mc,max(3300,lmax11,lmax12,lmax22,10000))
  
  
  root_grp,hf = php.baseCreateParobject(pars.res_object)
  hascl = [0]*6
  hascl[0] = 1
  hascl = nm.array(hascl,dtype=nm.int)
  
    
  lkl_grp = php.add_lkl_generic(root_grp,"actspt",1,hascl,lmax,lmin)
  
  print "lmin11 = ",lmin11
  print "lmin12 = ",lmin12
  print "lmin22 = ",lmin22
  print "lmax11 = ",lmax11
  print "lmax12 = ",lmax12
  print "lmax22 = ",lmax22
  print "tt_lmax_mc = ",tt_lmax_mc
  
  lkl_grp.attrs["lmin11"] = lmin11
  lkl_grp.attrs["lmin12"] = lmin12
  lkl_grp.attrs["lmin22"] = lmin22
  lkl_grp.attrs["lmax11"] = lmax11
  lkl_grp.attrs["lmax12"] = lmax12
  lkl_grp.attrs["lmax22"] = lmax22
  
  lkl_grp.attrs["tt_lmax_mc"] = tt_lmax_mc
  
  lkl_grp.attrs["use_act_south"] = pars.int(default=1).use_act_south
  lkl_grp.attrs["use_act_equa"] = pars.int(default=1).use_act_equa
  lkl_grp.attrs["use_spt_highell"] = pars.int(default=1).use_spt_highell
  
  php.add_external_data(osp.realpath(pars.actspt_data),lkl_grp,tar=bool(pars.int(default=1).include))

  #lkl_grp.attrs["external_dir"] = osp.realpath(pars.actspt_data)
  hf.close()
  
  #if hasattr(clik,"clik"):
  #  res = php.add_selfcheck(pars.res_object,mcl)
  #  print "lkl for init cl %g"%res
  #
  #if "cl_save" in pars:
  #  f=open(pars.cl_save,"w")
  #  for ci in mcl:
  #    print >>f,ci
  #  f.close()

import sys
if __name__=="__main__":
  main(sys.argv)