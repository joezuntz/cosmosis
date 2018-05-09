from __future__ import print_function
import os

dirname,_=os.path.split(__file__)
libname=os.path.join(dirname,"experimental_fault_handler.so")

def activate_experimental_fault_handling():
    try:
        import faulthandler
    except ImportError:
        raise ImportError("You must install the python faulthandler package to use experimental fault handling")
    import ctypes
    lib = ctypes.cdll.LoadLibrary(libname)
    lib.enable_combined_segfault_handler.argtypes=[]
    lib.enable_combined_segfault_handler.restype = None
    lib.enable_combined_segfault_handler()
    print("Experimental fault handling enabled")