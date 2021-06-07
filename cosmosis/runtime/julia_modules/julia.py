import ctypes
import sys
import os




class JuliaModuleInfo(ctypes.Structure):
    _fields_ = [("setup", ctypes.c_voidp),
                ("execute", ctypes.c_voidp),
                ("cleanup", ctypes.c_voidp),
                ("name", ctypes.c_char_p),
                ]

JuliaModuleInfoPtr = ctypes.POINTER(JuliaModuleInfo)

class JuliaModule(object):
    lib = None
    def __init__(self, filepath):
        dirname, filename = os.path.split(filepath)
        module_name = filename[:-3]
        self.setup_library()
        print("Loading module")
        self.info = self.lib.load_module(dirname.encode('ascii'), module_name.encode('ascii'))
        if not self.info:
            raise ValueError("Error loading Julia module at {}".format(filepath))

    @classmethod
    def setup_library(cls):
        if cls.lib is not None:
            return

        libdir = os.path.split(__file__)[0]
        libname = os.path.join(libdir, "libcosmosis_julia.so")

        if not os.path.exists(libname):
            raise RuntimeError("You must compile cosmosis with Julia support by running make in {}".format(libdir))
        cls.lib = ctypes.CDLL(libname, mode=ctypes.RTLD_GLOBAL)

        cls.lib.load_module.restype = JuliaModuleInfoPtr
        cls.lib.load_module.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

        cls.lib.run_setup.argtypes = [JuliaModuleInfoPtr, ctypes.c_voidp]
        cls.lib.run_setup.restype = ctypes.c_voidp

        cls.lib.run_execute.argtypes = [JuliaModuleInfoPtr, ctypes.c_voidp, ctypes.c_voidp]
        cls.lib.run_execute.restype = ctypes.c_int

        cls.lib.run_cleanup.argtypes = [JuliaModuleInfoPtr, ctypes.c_voidp]
        cls.lib.run_cleanup.restype = ctypes.c_int


    def setup(self, options_ptr):
        config = self.lib.run_setup(self.info, options_ptr)
        if not config:
            raise ValueError("Error running Julia setup function - backtrace above")
        return config

    def execute(self, block_ptr, config_ptr):
        return self.lib.run_execute(self.info, block_ptr, config_ptr)

    def cleanup(self, config_ptr):
        return self.lib.run_cleanup(self.info, config_ptr)

