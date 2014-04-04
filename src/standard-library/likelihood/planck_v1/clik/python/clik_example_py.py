#! PYTHONEXE
import sys
sys.path = ["REPLACEPATH"]+sys.path

import numpy as nm
import clik

def main(argv):
  if clik.try_lensing(argv[1]):
    main_lensing(argv)
    return 
  main_CMB(argv)

def main_CMB(argv):
  lklfile = argv[1]
  lkl = clik.clik(lklfile)
  for clfile in argv[2:]:
    cls = nm.loadtxt(clfile)
    nres = lkl(cls.flat[:])
    print nres

def main_lensing(argv):
  lklfile = argv[1]
  lkl = clik.clik_lensing(lklfile)
  for clfile in argv[2:]:
    cls = nm.loadtxt(clfile)
    nres = lkl(cls.flat[:])
    print nres

if __name__=="__main__":
  main(sys.argv)