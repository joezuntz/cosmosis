#! PYTHONEXE

import sys
sys.path = ["REPLACEPATH"]+sys.path

import numpy as nm
import clik
import clik.hpy as hpy

def main(argv):
  if len(sys.argv)!=2:
    print "usage : %s lkl_file"
    sys.exit(1)
    
  lkls = hpy.File(sys.argv[1])["clik"]
  print "found %d likelihoods"%lkls.attrs["n_lkl_object"]
  f0 = sys.argv[1]
  f_tmpl = f0.split(".")
  f_tmpl = ".".join(f_tmpl[:-1]+["%s"]+[f_tmpl[-1]])
  
  for lkln in ("lkl_%d"%v for v in range(lkls.attrs["n_lkl_object"])):
    fname = f_tmpl%lkln
    lkl = lkls[lkln]
    print "  "+fname
    hf = hpy.File(fname,"w",lkls)
    if "lmax" in lkl.attrs:
      lmax = lkl.attrs["lmax"]
    else:
      ell = lkl.attrs["ell"]
      lmax = nm.max(ell)      

    lmaxs = nm.where(lkl.attrs["has_cl"],lmax,-nm.ones(6,dtype=nm.int))
    chf = hf.create_group("clik")
    chf.attrs["lmax"] = lmaxs
    chf.attrs["n_lkl_object"] = 1
    lkl.copy(lkl,chf,"lkl_0")
    
    hf.close()
    
if __name__=="__main__":
  main(sys.argv)