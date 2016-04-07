import os
import ctypes
import sys
import string
import numpy as np
import time
import collections
import ConfigParser

import utils
import config
import parameter
import prior
import module
from cosmosis.datablock.cosmosis_py import block
import cosmosis.datablock.cosmosis_py as cosmosis_py


PIPELINE_INI_SECTION = "pipeline"

class MissingLikelihoodError(Exception):
    def __init__(self, message, data):
        super(MissingLikelihoodError, self).__init__(message)
        self.pipeline_data = data

class Pipeline(object):
    def __init__(self, arg=None, load=True):
        """ Initialize with a single filename or a list of them,
            a ConfigParser, or nothing for an empty pipeline"""
        if arg is None:
            arg = list()

        if isinstance(arg, config.Inifile):
            self.options = arg
        else:
            self.options = config.Inifile(arg)

        #This will be set later
        self.root_directory = self.options.get("runtime", "root", "cosmosis_none_signifier")
        if self.root_directory=="cosmosis_none_signifier":
            self.root_directory=None

        self.quiet = self.options.getboolean(PIPELINE_INI_SECTION, "quiet", True)
        self.debug = self.options.getboolean(PIPELINE_INI_SECTION, "debug", False)
        self.timing = self.options.getboolean(PIPELINE_INI_SECTION, "timing", False)
        shortcut = self.options.get(PIPELINE_INI_SECTION, "shortcut", "")
        if shortcut=="": shortcut=None

        # initialize modules
        self.modules = []
        if load and PIPELINE_INI_SECTION in self.options.sections():
            module_list = self.options.get(PIPELINE_INI_SECTION,
                                           "modules", "").split()

            for module_name in module_list:
                # identify module file

                filename = self.find_module_file(
                    self.options.get(module_name, "file"))

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
                                                  self.root_directory))
            self.shortcut_module=0
            self.shortcut_data=None
            if shortcut is not None:
                try:
                    index = module_list.index(shortcut)
                except ValueError:
                    raise ValueError("You tried to set a shortcut in "
                        "the pipeline but I do not know module %s"%shortcut)
                if index == 0:
                    print "You set a shortcut in the pipeline but it was the first module."
                    print "It will make no difference."
                self.shortcut_module = index

    def base_directory(self):
        if self.root_directory is None:
            try:
                self.root_directory = os.environ["COSMOSIS_SRC_DIR"]
                print "Root directory is ", self.root_directory
            except KeyError:
                self.root_directory = os.getcwd()
                print "WARNING: Could not find environment variable"
                print "COSMOSIS_SRC_DIR. Module paths assumed to be relative"
                print "to current directory, ", self.root_directory
        return self.root_directory

    def find_module_file(self, path):
        """Find a module file, which is assumed to be 
        either absolute or relative to COSMOSIS_SRC_DIR"""
        return os.path.join(self.base_directory(), path)

    def setup(self):
        if self.timing:
            timings = [time.time()]

        for module in self.modules:
            # identify parameters needed for module setup
            relevant_sections = [PIPELINE_INI_SECTION,
                                 "general",
                                 "logging",
                                 "debug",
                                 module.name]

            #We let the user specify additional global sections that are
            #visible to all modules
            global_sections = self.options.get("runtime", "global", " ")
            for global_section in global_sections.split():
                relevant_sections.append(global_section)


            config_block = block.DataBlock()

            for (section, name), value in self.options:
                if section in relevant_sections:
                    # add back a default section?
                    val = self.options.gettyped(section, name)
                    if val is not None:
                        config_block.put(section, name, val)

            module.setup(config_block, quiet=self.quiet)

            if self.timing:
                timings.append(time.time())

        if not self.quiet:
            sys.stdout.write("Setup all pipeline modules\n")

        if self.timing:
            timings.append(time.time())
            sys.stdout.write("Module timing:\n")
            for name, t2, t1 in zip(self.modules, timings[1:], timings[:-1]):
                sys.stdout.write("%s %f\n" % (name, t2-t1))

    def cleanup(self):
        for module in self.modules:
            module.cleanup()

    def make_graph(self, data, filename):
        try:
            import pygraphviz as pgv
        except ImportError:
            print "Cannot generate a graphical pipeline; please install the python package pygraphviz (e.g. with pip install pygraphviz)"
            return
        P = pgv.AGraph(directed=True)
        # P = pydot.Cluster(label="Pipeline", color='black',  style='dashed')
        # G.add_subgraph(P)
        def norm_name(name):
            return name #.replace("_", " ").title()
        P.add_node("Sampler", color='Pink', style='filled', group='pipeline',shape='octagon', fontname='Courier')
        for module in self.modules:
            # module_node = pydot.Node(module.name, color='Yellow', style='filled')
            P.add_node(norm_name(module.name), color='lightskyblue', style='filled', group='pipeline', shape='box')
        P.add_edge("Sampler", norm_name(self.modules[0].name), color='lightskyblue', style='bold', arrowhead='none')
        for i in xrange(len(self.modules)-1):
            P.add_edge(norm_name(self.modules[i].name),norm_name(self.modules[i+1].name), color='lightskyblue', style='bold', arrowhead='none')
        # D = pydot.Cluster(label="Data", color='red', style='dashed')
        # G.add_subgraph(D)
        # #find
        log = [data.get_log_entry(i) for i in xrange(data.get_log_count())]
        known_sections = set()
        for entry in log:
            if entry!="MODULE-START":
                section = entry[1]
                if section not in known_sections:
                    if section=="Results":
                        P.add_node(norm_name(section), color='Pink', style='filled', shape='octagon', fontname='Courier')
                    else:                        
                        P.add_node(norm_name(section), color='yellow', style='filled', fontname='Courier', shape='box')
                    known_sections.add(section)
        module="Sampler"
        known_edges = set()
        for entry in log:
            if entry[0]=="MODULE-START":
                module=entry[1]
            elif entry[0]=="WRITE-OK" or entry[0]=="REPLACE-OK":
                section=entry[1]
                if (module,section,'write') not in known_edges:
                    P.add_edge(norm_name(module), norm_name(section), color='green')
                    known_edges.add((module,section,'write'))
            elif entry[0]=="READ-OK":
                section=entry[1]
                if (section,module,'read') not in known_edges:
                    P.add_edge((norm_name(section),norm_name(module)), color='grey50')
                    known_edges.add((section,module,'read'))

        P.write(filename)

    def run(self, data_package):
        modules = self.modules
        first = (self.shortcut_data is None)
        if self.shortcut_module and not first:
            modules = modules[self.shortcut_module:]

        for module_number, module in enumerate(modules):
            if self.debug:
                sys.stdout.write("Running %.20s ...\n" % module)
                sys.stdout.flush()
            data_package.log_access("MODULE-START", module.name, "")
            if self.timing:
                t1 = time.time()

            status = module.execute(data_package)

            if self.debug:
                sys.stdout.write("Done %.20s status = %d \n" % (module,status))
                sys.stdout.flush()

            if self.timing:
                t2 = time.time()
                sys.stdout.write("%s took: %f seconds\n"% (module,t2-t1))

            if status:
                if self.debug:
                    data_package.print_log()
                    sys.stdout.flush()
                    sys.stderr.write("Because you set debug=True I printed a log of "
                                     "all access to data printed above.\n"
                                     "Look for the word 'FAIL' \n")
                    sys.stderr.write("Though the error message could also be somewhere above that.\n\n")
                if not self.quiet:
                    sys.stderr.write("Error running pipeline (%d)- "
                                     "hopefully printed above here.\n"%status)
                    sys.stderr.write("Aborting this run and returning "
                                     "error status.\n")
                    if not self.debug:
                        sys.stderr.write("Setting debug=T in [pipeline] might help.\n")
                return None

            if self.shortcut_module and first and module_number==self.shortcut_module-1:
                print "Saving shortcut data"
                self.shortcut_data = data_package.clone()


        if not self.quiet:
            sys.stdout.write("Pipeline ran okay.\n")

        data_package.log_access("MODULE-START", "Results", "")
        # return something
        return True


class LikelihoodPipeline(Pipeline):
    def __init__(self, arg=None, id="",override=None, load=True):
        super(LikelihoodPipeline, self).__init__(arg=arg, load=load)

        if id:
            self.id_code = "[%s] " % str(id)
        else:
            self.id_code = ""
        self.n_iterations = 0

        values_file = self.options.get(PIPELINE_INI_SECTION, "values")
        self.values_filename=values_file
        priors_files = self.options.get(PIPELINE_INI_SECTION,
                                        "priors", "").split()
        self.priors_files = priors_files

        self.parameters = parameter.Parameter.load_parameters(values_file,
                                                              priors_files,
                                                              override,
                                                              )

        self.reset_fixed_varied_parameters()

        #We want to save some parameter results from the run for further output
        extra_saves = self.options.get(PIPELINE_INI_SECTION,
                                       "extra_output", "")

        self.extra_saves = []
        for extra_save in extra_saves.split():
            section, name = extra_save.upper().split('/')
            self.extra_saves.append((section, name))

        self.number_extra = len(self.extra_saves)
        #pull out all the section names and likelihood names for later
        self.likelihood_names = self.options.get(PIPELINE_INI_SECTION,
                                                 "likelihoods").split()

        # now that we've set up the pipeline properly, initialize modules
        self.setup()

    def reset_fixed_varied_parameters(self):
        self.varied_params = [param for param in self.parameters
                              if param.is_varied()]
        self.fixed_params = [param for param in self.parameters
                             if param.is_fixed()]     

    def parameter_index(self, section, name):
        i = self.parameters.index((section, name))
        if i==-1:
            raise ValueError("Could not find index of parameter %s in section %s"%(name, section))
        return i

    def set_varied(self, section, name, lower, upper):
        i = self.parameter_index(section, name)
        self.parameters[i].limits = (lower,upper)
        self.reset_fixed_varied_parameters()

    def set_fixed(self, section, name, value):
        i = self.parameter_index(section, name)
        self.parameters[i].limits = (value, value)
        self.reset_fixed_varied_parameters()


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

    def normalize_matrix(self, c):
        c = c.copy()
        n = c.shape[0]
        assert n==c.shape[1], "Cannot normalize a non-square matrix"
        for i in xrange(n):
            pi = self.varied_params[i]
            ri = pi.limits[1] - pi.limits[0]
            for j in xrange(n):
                pj = self.varied_params[j]
                rj = pj.limits[1] - pj.limits[0]
                c[i,j] /= (ri*rj)
        return c

    def denormalize_matrix(self, c, inverse=False):
        c = c.copy()
        n = c.shape[0]
        assert n==c.shape[1], "Cannot normalize a non-square matrix"
        for i in xrange(n):
            pi = self.varied_params[i]
            ri = pi.limits[1] - pi.limits[0]
            for j in xrange(n):
                pj = self.varied_params[j]
                rj = pj.limits[1] - pj.limits[0]
                if inverse:
                    c[i,j] /= (ri*rj)
                else:
                    c[i,j] *= (ri*rj)
        return c


    def start_vector(self, all_params=False):
        if all_params:
            return np.array([param.start for
                 param in self.parameters])
        else:            
            return np.array([param.start for
                         param in self.varied_params])

    def min_vector(self, all_params=False):
        if all_params:
            return np.array([param.limits[0] for
                 param in self.parameters])
        else:
            return np.array([param.limits[0] for
                         param in self.varied_params])

    def max_vector(self, all_params=False):
        if all_params:
            return np.array([param.limits[1] for
                 param in self.parameters])
        else:
            return np.array([param.limits[1] for
                         param in self.varied_params])


    def run_parameters(self, p, check_ranges=False, all_params=False):
        if check_ranges:
            if self.is_out_of_range(p):
                return None

        if self.shortcut_module and self.shortcut_data is not None:
            data = self.shortcut_data.clone()
        else:
            data = block.DataBlock()

        if all_params:
            for param, x in zip(self.parameters, p):
                data[param.section, param.name] = x
        else:
            # add varied parameters
            for param, x in zip(self.varied_params, p):
                data[param.section, param.name] = x

            # add fixed parameters
            for param in self.fixed_params:
                data[param.section, param.name] = param.start

        if self.run(data):
            return data
        else:
            return None

    def create_ini(self, p, filename):
        "Dump the specified parameters as a new ini file"
        output = collections.defaultdict(list)
        for param, x in zip(self.varied_params, p):
            output[param.section].append("%s  =  %r    %r    %r\n" % (
                param.name, param.limits[0], x, param.limits[1]))
        for param in self.fixed_params:
            output[param.section].append("%s  =  %r\n" % (param.name, param.start))
        ini = open(filename, 'w')
        for section, params in sorted(output.items()):
            ini.write("[%s]\n"%section)
            for line in params:
                ini.write(line)
            ini.write("\n")
        ini.close()


    def prior(self, p, all_params=False):
        if all_params:
            params = self.parameters
        else:
            params = self.varied_params
        return sum([param.evaluate_prior(x) for param, x in
                    zip(params, p)])

    def posterior(self, p, return_data=False, all_params=False):
        prior = self.prior(p, all_params=all_params)
        if prior == -np.inf:
            if not self.quiet:
                sys.stdout.write("Proposed outside bounds\nPrior -infinity\n")
            if return_data:
                return prior, np.repeat(np.nan, self.number_extra), None
            return prior, np.repeat(np.nan, self.number_extra)
        if return_data:
            like, extra, data = self.likelihood(p, return_data=True, all_params=all_params)
            return prior + like, extra, data
        else:
            like, extra = self.likelihood(p, all_params=all_params)
            return prior + like, extra
        
    def likelihood(self, p, return_data=False, all_params=False):
        #Set the parameters by name from the parameter vector
        #If one is out of range then return -infinity as the log-likelihood
        #i.e. likelihood is zero.  Or if something else goes wrong do the same
        data = self.run_parameters(p, all_params=all_params)
        if data is None:
            if return_data:
                return -np.inf, np.repeat(np.nan, self.number_extra), data
            else:
                return -np.inf, np.repeat(np.nan, self.number_extra)

        # loop through named likelihoods and sum their values
        likelihoods = []
        section_name = cosmosis_py.section_names.likelihoods
        for likelihood_name in self.likelihood_names:
            try:
                L = data.get_double(section_name,likelihood_name+"_like")
                likelihoods.append(L)
            except block.BlockError:
                raise MissingLikelihoodError(likelihood_name, data)

        like = sum(likelihoods)

        if not self.quiet and self.likelihood_names:
            sys.stdout.write("Likelihood %e\n" % (like,))

        extra_saves = []
        for option in self.extra_saves:
            try:
                #JAZ - should this be just .get(*option) ?
                value = data.get_double(*option)
            except block.BlockError:
                value = np.nan

            extra_saves.append(value)

        self.n_iterations += 1
        if return_data:
            return like, extra_saves, data
        else:
            return like, extra_saves
