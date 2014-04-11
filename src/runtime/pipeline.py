import os
import ctypes
import sys
import string
import numpy as np
import time
import ConfigParser

import utils
import config
import parameter
import prior
import module
from cosmosis_py import block
import cosmosis_py.section_names


PIPELINE_INI_SECTION = "pipeline"


class Pipeline(object):
    def __init__(self, arg=None, quiet=True, debug=False, timing=False):
        """ Initialize with a single filename or a list of them,
            a ConfigParser, or nothing for an empty pipeline"""
        if arg is None:
            arg = list()

        if isinstance(arg, config.Inifile):
            self.options = arg
        else:
            self.options = config.Inifile(arg)

        self.quiet = quiet
        self.debug = debug
        self.timing = timing

        # initialize modules
        self.modules = []
        if PIPELINE_INI_SECTION in self.options.sections():
            rootpath = self.options.get(PIPELINE_INI_SECTION,
                                        "root",
                                        os.curdir)
            module_list = self.options.get(PIPELINE_INI_SECTION,
                                           "modules", "").split()

            for module_name in module_list:
                # identify module file
                filename = self.options.get(module_name, "file")

                # identify relevant functions
                setup_function = self.options.get(module_name,
                                                  "setup", "setup")
                exec_function = self.options.get(module_name,
                                                 "function", "execute")
                cleanup_function = self.options.get(module_name,
                                                    "cleanup", "cleanup")

                self.modules.append(module.Module(module_name,
                                                  filename,
                                                  setup_function,
                                                  exec_function,
                                                  cleanup_function,
                                                  rootpath))

    def setup(self):
        if self.timing:
            timings = [time.clock()]

        for module in self.modules:
            # identify parameters needed for module setup
            relevant_sections = [PIPELINE_INI_SECTION,
                                 "general",
                                 "logging",
                                 "debug",
                                 module.name]

            config_block = block.DataBlock()

            for (section, name), value in self.options:
                if section in relevant_sections:
                    # add back a default section?
                    config_block.put(section, name,
                                     self.options.gettyped(section, name))

            module.setup(config_block)

            if self.timing:
                timings.append(time.clock())

        if not self.quiet:
            sys.stdout.write("Setup all pipeline modules\n")

        if self.timing:
            timings.append(time.clock())
            sys.stdout.write("Module timing:\n")
            for name, t2, t1 in zip(self.modules, timings[1:], timings[:-1]):
                sys.stdout.write("%s %f\n" % (name, t2-t1))

    def cleanup(self):
        for module in self.modules:
            module.cleanup()

    def run(self, data_package):
        if self.timing:
            timings = [time.clock()]

        for module in self.modules:
            if self.debug:
                sys.stdout.write("Running %.20s ... " % module)
                sys.stdout.flush()

            status = module.execute(data_package)

            if self.timing:
                timings.append(time.clock())

            if status:
                if not self.quiet:
                    sys.stderr.write("Error running pipeline (%d)- "
                                     "hopefully printed above here.\n")
                    sys.stderr.write("Aborting this run and returning "
                                     "error status.\n")

                    if self.timing:
                        sys.stdout.write("Module timing:\n")
                        for name, t2, t1 in zip(self.module_names[:],
                                                timings[1:], timings[:-1]):
                            sys.stdout.write("%s %f\n" % (name, t2-t1))
                return None

        if not self.quiet:
            sys.stdout.write("Pipeline ran okay.\n")

        if self.timing:
            sys.stdout.write("Module timing:\n")
            for name, t2, t1 in zip(self.modules, timings[1:], timings[:-1]):
                sys.stdout.write("%s %f\n" % (name, t2-t1))

        # return something
        return True


class LikelihoodPipeline(Pipeline):
    def __init__(self, arg=None, id="", debug=False,
                 quiet=True, timing=False):
        super(LikelihoodPipeline, self).__init__(arg=arg, quiet=quiet,
                                                 debug=debug, timing=timing)

        if id:
            self.id_code = "[%s] " % str(id)
        else:
            self.id_code = ""
        self.n_iterations = 0

        values_file = self.options.get(PIPELINE_INI_SECTION, "values")
        priors_files = self.options.get(PIPELINE_INI_SECTION,
                                        "priors", "").split()

        self.parameters = parameter.Parameter.load_parameters(values_file,
                                                              priors_files)

        self.varied_params = [param for param in self.parameters
                              if param.is_varied()]
        self.fixed_params = [param for param in self.parameters
                             if param.is_fixed()]

        #We want to save some parameter results from the run for further output
        extra_saves = self.options.get(PIPELINE_INI_SECTION,
                                       "extra_output", "")

        self.extra_saves = []
        for extra_save in extra_saves.split():
            section, name = extra_save.upper().split('/')
            if not (section, name) in parameters:
                raise RuntimeError()
            self.extra_saves.append((section, name))

        #pull out all the section names and likelihood names for later
        self.likelihood_names = self.options.get(PIPELINE_INI_SECTION,
                                                 "likelihoods").split()

        # now that we've set up the pipeline properly, initialize modules
        self.setup()

    def output_names(self):
        param_names = [str(p) for p in self.varied_params]
        extra_names = ['%s--%s'%p for p in self.extra_saves]
        return param_names + extra_names

    def randomized_start(self):
        # should have different randomization strategies
        # (uniform, gaussian) possibly depending on prior?
        return np.array([p.random_point() for p in self.varied_params])

    def is_out_of_range(self, p):
        return any([not param.in_range(x) for
                    param, x in zip(self.varied_params, p)])

    def denormalize_vector(self, p):
        return np.array([param.denormalize(x) for param, x
                         in zip(self.varied_params, p)])

    def normalize_vector(self, p):
        return np.array([param.normalize(x) for param, x
                         in zip(self.varied_params, p)])

    def start_vector(self, p):
        return np.array([param.start for
                         param in self.varied_params])

    def run_parameters(self, p, check_ranges=False):
        if check_ranges:
            if self.is_out_of_range(params_by_section):
                return None

        data = block.DataBlock()

        # add varied parameters
        for param, x in zip(self.varied_params, p):
            data.put_double(param.section, param.name, x)

        # add fixed parameters
        for param in self.fixed_params:
            data.put_double(param.section, param.name, param.start)

        if self.run(data):
            return data
        else:
            return None

    def prior(self, p):
        return sum([param.evaluate_prior(x) for param, x in
                    zip(self.varied_params, p)])

    def posterior(self, p):
        prior = self.prior(p)
        if prior == -np.inf:
            return prior, utils.everythingIsNan
        like, extra = self.likelihood(p)
        return prior + like, extra

    def likelihood(self, p, return_data=False):
        #Set the parameters by name from the parameter vector
        #If one is out of range then return -infinity as the log-likelihood
        #i.e. likelihood is zero.  Or if something else goes wrong do the same
        data = self.run_parameters(p)
        if data is None:
            if return_data:
                return -np.inf, utils.everythingIsNan, data
            else:
                return -np.inf, utils.everythingIsNan

        # loop through named likelihoods and sum their values
        try:
            like = sum([data.get_double(cosmosis_py.section_names.likelihoods,
                                        likelihood_name+"_like")
                        for likelihood_name in self.likelihood_names])
        except block.BlockError:
            if return_data:
                return -np.inf, utils.everythingIsNan, data
            else:
                return -np.inf, utils.everythingIsNan

        if not self.quiet:
            sys.stdout.write("Likelihood %e\n" % (like,))

        extra_saves = {}
        for option in self.extra_saves:
            try:
                #JAZ - should this be just .get(*option) ?
                value = data.get_double(*option)
            except BlockError:
                value = np.nan

            extra_saves[option] = value
        extra_saves['LIKE'] = like

        self.n_iterations += 1
        if return_data:
            return like, extra_saves, data
        else:
            return like, extra_saves
