import os
import ctypes
import sys
import numpy as np

from cosmosis.datablock import option_section

MODULE_TYPE_EXECUTE_SIMPLE = "execute"
MODULE_TYPE_EXECUTE_CONFIG = "execute_config"
MODULE_TYPE_SETUP = "setup"
MODULE_TYPE_CLEANUP = "cleanup"

MODULE_LANG_PYTHON = "python"
MODULE_LANG_DYLIB = "dylib"


class SetupError(Exception):
    pass


class Module(object):
    def __init__(self, module_name, file_path,
                 setup_function, execute_function,
                 cleanup_function, rootpath="."):

        self.name = module_name

        self.setup_function = setup_function
        self.execute_function = execute_function
        self.cleanup_function = cleanup_function

        # identify module filename
        filename = file_path
        if not os.path.isabs(filename):
            filename = os.path.join(rootpath, filename)
        self.filename = filename

        self.library, language = Module.load_library(filename)
        self.is_python = (language == MODULE_LANG_PYTHON)

        # attempt to load setup and cleanup functions
        self.setup_function = Module.load_function(self.library,
                                                   setup_function,
                                                   MODULE_TYPE_SETUP)
        self.cleanup_function = Module.load_function(self.library,
                                                     cleanup_function,
                                                     MODULE_TYPE_CLEANUP)

    def copy_section_to_module_options(self, config):
        if config.has_section(option_section):
            config._delete_section(option_section)
        for (section, name) in config.keys(self.name):
            config[option_section, name] = config[section, name]

    def setup(self, config, quiet=True):
        self.copy_section_to_module_options(config)
        if not self.is_python:
            config = config._ptr

        if self.setup_function:
            if not quiet:
                print '-- Setting up module %s --' % (self.name)
            self.data = self.setup_function(config)
        else:
            self.data = None

        if self.data is not None:
            module_type = MODULE_TYPE_EXECUTE_CONFIG
        else:
            module_type = MODULE_TYPE_EXECUTE_SIMPLE

        self.execute_function = Module.load_function(self.library,
                                                     self.execute_function,
                                                     module_type)
        if self.execute_function is None:
            raise ValueError("Could not find a function 'execute' in module %s"%self.name)

    def execute(self, data_block):
        if not self.is_python:
            data_block = data_block._ptr
        if self.data is not None:
            return self.execute_function(data_block, self.data)
        else:
            return self.execute_function(data_block)

    def cleanup(self):
        if self.cleanup_function:
            self.cleanup_function(self.data)

    def __str__(self):
        return self.name

    @staticmethod
    def load_library(filepath):
        if filepath.endswith('so') or filepath.endswith('dylib'):
            language = MODULE_LANG_DYLIB
            try:
                library = ctypes.cdll.LoadLibrary(filepath)
            except OSError as error:
                exists = os.path.exists(filepath)
                if exists:
                    raise SetupError("You specified a path %s for a module. "
                                     "File exists, but could not be opened. "
                                     "Error was %s" % (filepath, error))
                else:
                    raise SetupError("You specified a path %s for a module. "
                                     "File does not exist.  Error was %s" %
                                     (filepath, error))
        else:
            language = MODULE_LANG_PYTHON
            dirname, filename = os.path.split(filepath)
            # allows .pyc and .py modules to be used
            impname, ext = os.path.splitext(filename)
            sys.path.insert(0, dirname)
            try:
                library = __import__(impname)
            except ImportError as error:
                raise SetupError("You specified a path %s for a module. "
                                 "I looked for a python module there but "
                                 "was unable to load it.  Error was %s" %
                                 (filepath, error))
            sys.path.pop(0)

        return library, language

    @staticmethod
    def load_function(library, function_name,
                      module_type=MODULE_TYPE_EXECUTE_SIMPLE):
        "Load a module from a shared library"
        function = getattr(library, function_name, None)
        if not function:
            function = getattr(library, function_name + "_", None)

        if function:
            if module_type == MODULE_TYPE_EXECUTE_SIMPLE:
                function.argtypes = [ctypes.c_voidp]
                function.restype = ctypes.c_int
            elif module_type == MODULE_TYPE_EXECUTE_CONFIG:
                function.argtypes = [ctypes.c_voidp, ctypes.c_voidp]
                function.restype = ctypes.c_int
            elif module_type == MODULE_TYPE_SETUP:
                function.argtypes = [ctypes.c_voidp]
                function.restype = ctypes.c_voidp
            elif module_type == MODULE_TYPE_CLEANUP:
                function.argtypes = [ctypes.c_voidp]
                function.restype = ctypes.c_int
            else:
                raise ValueError("Unknown module type passed "
                                 "to load_interface")
        return function


    @classmethod
    def from_options(cls,module_name,options,root_directory=None):
        if root_directory is None:
            root_directory = os.environ.get("COSMOSIS_SRC_DIR", ".")

        filename = cls.find_module_file(root_directory,
            options.get(module_name, "file"))

        # identify relevant functions
        setup_function = options.get(module_name,
                                      "setup", "setup")
        exec_function = options.get(module_name,
                                     "function", "execute")
        cleanup_function = options.get(module_name,
                                        "cleanup", "cleanup")

        m = cls(module_name, filename,
              setup_function, exec_function, cleanup_function,
              root_directory)

        return m

    @staticmethod
    def find_module_file(base_directory, path):
        """Find a module file, which is assumed to be 
        either absolute or relative to COSMOSIS_SRC_DIR"""
        return os.path.join(base_directory, path)
