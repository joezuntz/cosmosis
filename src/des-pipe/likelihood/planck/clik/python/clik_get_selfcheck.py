#! PYTHONEXE
import sys
sys.path = ["REPLACEPATH"]+sys.path

import clik.parobject as php
import clik
import clik.hpy as hpy



def main(argv):
  
  lkl = hpy.File(sys.argv[1],"r")
  cls = lkl["clik/check_param"][:]
  if len(argv)==2:
  	clfile = argv[1]+".cls"
  else:
  	clfile = argv[2]
  cls.tofile(clfile,sep=" ")
    
import sys
if __name__=="__main__":
  main(sys.argv)