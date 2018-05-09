#coding: utf-8

u"""Definition of :class:`Module`."""

from __future__ import print_function
from builtins import object

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

    u"""Tenuous distinction between generic error and :class:`Module` configuration error.

    :class:`SetupError`, :class:`ValueError` and the generic
    :class:`Exception` should be regarded as synonyms in the context of
    the :class:`Module` class: all indicate some basic problem in the
    .ini files.
"""

    pass



class Module(object):

    u"""Interface to user-defined components of computational pipelines.

    To alleviate confusion, distinguish in your mind the notion of
    :class:`Module` as a cosmosis Module (capital ‘M’ for distinction) as
    opposed to a Python module!

    A :class:`Module` represents a single discrete step in an overall 
    computational pipeline, consuming some inputs from and providing some
    outputs to a :class:`DataBlock`.

    :class:`Module`s are made from either a python file or a shared library,
    and must have a setup function, to be run once when the module is first
    created, and an execute function, to be run whenever a new calculation 
    is to be computed.
    
    This Python class manages the lifetime of a Module, 
    and provides a high-level interface to the module to the rest of the application.

    Unless you are doing something fairly complicated you are unlikely to use
    a module in your own scripts - more generally you would use a :class:`Pipeline`,
    which collects a sequence of modules together.

    The Module /setup/ function, if present, takes in a configuration
    :class:`DataBlock` object (it will find the moduleʼs parameters in a “module_options”
    section), and may return some object which will subsequently be passed
    to the Moduleʼs /execute/ function.

    The /execute/ function itself (which MUST be present in the linked
    library) is also called with a :class:`DataBlock`, and if /setup/
    provided data back to the wrapper, then the /execute/ function MUST
    accept this type of object as its second argument.

    The optional /cleanup/ function is also passed the /setup/ʼs `data` object
    (only), and so may free any resources which that object clings on to.

    """

    def __init__(self, module_name, file_path,
                 setup_function="setup", execute_function="execute",
                 cleanup_function="cleanup", rootpath="."):
        u"""Create an object of type `module_name` from dynamic load library at `file_path`, with interface specified by the `*_function`s.

        The `rootpath` is the directory to search for the linkable
        library.

        Note how `self.*_function`s start out as strings and then become
        executable function objects as the initialization
        progresses—except for `execute_function which is not loaded until
        :func:`setup is called.

        """
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
        u"""Re-compose the ‘module_options’ section of `config`.

        Remove an existing ‘module_options’ section of `config`, and
        replace it by scanning the entire `config`uration for keys under
        a section named after us and copying their values into a new
        ‘module_options’ section.

        This is done to help the implementation of the module deal with
        its configuration in a generic way, i.e. all modules can simply
        refer to `module_options` for their parameters.

        """
        if config.has_section(option_section):
            config._delete_section(option_section)
        for (section, name) in config.keys(self.name):
            config[option_section, name] = config[section, name]



    def setup(self, config, quiet=True):
        u"""Call the /Module/ constructor.

        This method also pulls in the `execute_function` from the linked
        library.

        This function MUST be run after object initialization and before
        any other action takes place.

        """
        self.copy_section_to_module_options(config)
        if not self.is_python:
            config = config._ptr

        if self.setup_function:
            if not quiet:
                print('-- Setting up module %s --' % (self.name))
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
            raise ValueError("Could not find a function 'execute' in module '"
                                 +  self.name + "'")
        


    def execute(self, data_block):
        u"""Run the /execute/ function and return whatever it does.

        If the /setup/ function provided some data object, this will be
        passed to the /execute/ function as a second argument.

        """
        if not self.is_python:
            data_block = data_block._ptr
        if self.data is not None:
            return self.execute_function(data_block, self.data)
        else:
            return self.execute_function(data_block)



    def cleanup(self):
        u"""Run the /cleanup/ function.

        If the /setup/ function provided a data object, this will be
        passed to /cleanup/.

        """
        if self.cleanup_function:
            self.cleanup_function(self.data)



    def __str__(self):
        u"""Return the `name` of this Module."""
        return self.name



    @staticmethod
    def load_library(filepath):
        u"""Whatever kind of file is at `filepath`, try to load it into memory.

        This is really two different functions discriminated by the
        extension on `filepath`: ‘.so’ or ‘.dylib’ will be linked as a
        C-interfaced runtime loadable library, and anything else will be
        taken as a Python module.

        """

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
        u"""Load a Module's functions from a shared library."""
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
                raise ValueError("Unknown module type passed to load_interface")
        return function



    @classmethod
    def from_options(cls,module_name,options,root_directory=None):
        u"""The real class constructor.  Return a :class:`Module` based on the contents of user `options`.

        Use the contents of the `module_name` section of the `options` to
        construct a new Module wrapper.

        If `root_directory` (the place to look for Module implementation
        files and maybe associated static data files) is not specified,
        then either the environment variable ‘COSMOSIS_SRC_DIR’ will be
        used, or else the current working directory.

        """
        if root_directory is None:
            root_directory = os.environ.get("COSMOSIS_SRC_DIR", ".")

        filename = cls.find_module_file(root_directory,
                                        options.get(module_name, "file"))

        # identify relevant functions
        setup_function = options.get(module_name, "setup", fallback="setup")
        exec_function = options.get(module_name, "function", fallback="execute")
        cleanup_function = options.get(module_name, "cleanup", fallback="cleanup")

        m = cls(module_name, filename,
                setup_function, exec_function, cleanup_function,
                root_directory)

        return m



    @staticmethod
    def find_module_file(base_directory, path):
        u"""Find a module file, which is assumed to be either absolute or relative to COSMOSIS_SRC_DIR."""
        return os.path.join(base_directory, path)
