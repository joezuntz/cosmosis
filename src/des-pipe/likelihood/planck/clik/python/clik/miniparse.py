import re
import numpy as nm

def scarray(li,scal=False):
  if len(li)==1 and scal:
    return li[0]
  else:
    return nm.array(li)
    
class transformme:
  def __init__(self,tfunc,pf,isar=False):
    self.tfunc = tfunc
    self.pf = pf
    self.df = None
    self.isar = isar
    self.scal=False
    
  def __getattr__(self,val):
    try:
      vl = self.pf.pf[val]
    except Exception,e:
      if self.df==None:
        raise e
      else:
        vl = self.df
    self.pf._access_list[val] = vl
    if self.isar:
      if isinstance(vl,str):
        vvl = vl.split()
      else:
        vvl = vl
      return scarray([self.tfunc(v) for v in vvl],self.scal)
    return self.tfunc(vl)
    
  def __call__(self,**kk):
    if "default" in kk:
      self.df = kk["default"]
    if kk.get("scalarize",False):
      self.scal = True
    return self

class miniparse(object):
  def __init__(self, pfn,**kk):
    self.pf = {}
    
    if pfn!=None:
      print "read parameter file %s"%pfn
      pff =open(pfn)
      txt = "\n".join([to.split("#")[0] for to in pff])+"\n"
      pf = dict(re.findall("(?<!#)(\w+)\s*=\s*(.+?)\n",txt))
      self.pf.update(pf)
    
    self.pf.update(kk)
    self._access_list = {}
    
  def __repr__(self):
    rr = []
    for v in self._access_list:
      rr += ["%s = %s"%(v,self._access_list[v])]
    return "\n".join(rr)
    
  def __contains__(self,val):
    res = val in self.pf
    if res:
      self._access_list[val] = getattr(self,val)
    return res

  @property
  def bool(self):
    return transformme(lambda val:str(val).lower() in ("t","0","true"),self)

  @property
  def bool_array(self):
    return transformme(lambda val:str(val).lower() in ("t","0","true"),self,True)
    
  @property
  def int(self):
    return transformme(int,self)

  @property
  def int_array(self):
    return transformme(int,self,True)

  @property
  def float(self):
    return transformme(float,self)

  @property
  def float_array(self):
    return transformme(float,self,True)

  @property
  def str(self):
    return transformme(str,self)

  @property
  def str_array(self):
    return transformme(str,self,True)

  def __getattr__(self,val):
    res = getattr(self.str,val)
    return res
    
def fromargv():
  import sys
  argv = sys.argv

  if len(argv)!=2:
    print "usage: %s parfile\n"%(argv[0])
    sys.exit(-1)

  pfn = argv[1]
  pf = miniparse(pfn)
  return pf