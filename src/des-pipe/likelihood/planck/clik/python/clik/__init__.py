import os.path as osp

if osp.exists(osp.join(osp.dirname(__file__),"lkl.pyx")):
  raise ImportError("Cannot import clik python wrapper from the source directory.\nMake sure that you have compiled and installed clik and then\nrun python from another directory.")

try:
  from lkl import clik
  _lkl_ok = True
except ImportError,e:
  print "Cannot use clik wrapper (cause = '%s')"%e
try:
  from lkl_lensing import clik_lensing,try_lensing
  _lkl_lensing_ok = True
except ImportError,e:
  print "Cannot use clik_lensing wrapper (cause = '%s')"%e
  def try_lensing(fl):
    return False
  
import re
import numpy as nm

from miniparse import miniparse

class forfile:
  def __init__(self,fi):
    
    if isinstance(fi,file):
      self.fi = fi
    else :
      self.fi=open(fi)
    self.bf=""
  def read(self,fmt=''):
    if self.bf=='':
      sz = nm.fromstring(self.fi.read(4),dtype=nm.int32)[0]
      #print "want %d bytes"%sz
      self.bf = self.fi.read(sz)
      #print self.bf
      sz2 =nm.fromstring(self.fi.read(4),dtype=nm.int32)[0]
      #print sz2 
      assert sz==sz

    if fmt=='':
      self.bf=''
      return
    res = [self.cvrt(ff) for ff in fmt.strip().split()]
    if len(res)==1:
      return res[0]
    return tuple(res)
  
  def cvrt(self,fmt):
    cmd = re.findall("([0-9]*)([i|f])([0-9]+)",fmt)[0]
    dtype = nm.dtype({"f":"float","i":"int"}[cmd[1]]+cmd[2])
    itm = nm.array(1,dtype=dtype).itemsize
    nelem=1
    if cmd[0]: 
      nelem = int(cmd[0])
    res = nm.fromstring(self.bf[:itm*nelem],dtype=dtype)
    self.bf=self.bf[itm*nelem:]
    if nelem==1:
      return res[0]
    return res
  
  def close(self):
    self.bf=''
    self.fi.close()
  