#! PYTHONEXE
import sys
sys.path = ["REPLACEPATH"]+sys.path

import numpy as nm
import clik.parobject as php
import clik
import os.path as osp
import os
import clik.hpy as hpy

def main(argv):
  pars = clik.miniparse(argv[1])

  
  inhf = hpy.File(pars.input_object)
  
  if "external_dir" in inhf["clik/lkl_%d"%pars.int(default=0).lkl_id].attrs and inhf["clik/lkl_%d"%pars.int(default=0).lkl_id].attrs["external_dir"]==".":
    import shutil
    shutil.copytree(pars.input_object+"/clik/lkl_%d/_external"%pars.int(default=0).lkl_id,pars.install_path)
  else:
    dts = inhf["clik/lkl_%d/external_data"%pars.int(default=0).lkl_id][:]
    inhf.close()
    if not osp.exists(pars.install_path):
      os.mkdir(pars.install_path)
    f=open(osp.join(pars.install_path,"data.tar"),"w")
    f.write(dts.tostring())
    f.close()
    assert os.system("cd %s;tar xvf data.tar"%pars.install_path)==0
    assert os.system("cd %s;rm -f data.tar"%pars.install_path)==0
  
  if "res_object" in pars:
    hpy.copyfile(pars.input_object,pars.res_object)

    outhf = hpy.File(pars.res_object,"r+")
    try:
      del outhf["clik/lkl_%d/external_data"%pars.int(default=0).lkl_id]
    except Exception,e:
      pass
    try:
      rmtree(pars.input_object+"/clik/lkl_%d/_external"%pars.int(default=0).lkl_id)
    except Exception,e:
      pass
      
    outhf["clik/lkl_%d"%pars.int(default=0).lkl_id].attrs["external_dir"] = pars.install_path
    outhf.close()

import sys
if __name__=="__main__":
  main(sys.argv)