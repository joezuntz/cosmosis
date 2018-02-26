#coding: utf-8

u"""Definition of :class:`Pipeline` and the specialization :class:`LikelihoodPipeline`."""

from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import zip
from builtins import range
from builtins import object

import os
import ctypes
import sys
import string
import numpy as np
import time
import collections
import configparser
import traceback
import signal
from . import utils
from . import config
from . import parameter
from . import prior
from . import module
from cosmosis.datablock.cosmosis_py import block
import cosmosis.datablock.cosmosis_py as cosmosis_py
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass



PIPELINE_INI_SECTION = "pipeline"
NO_LIKELIHOOD_NAMES = "no_likelihood_names_sentinel"

class MissingLikelihoodError(Exception):

    u"""Class to throw if there are not enough data for a likelihood method.

    This class is specifically designed to be thrown during a
    `pipeline.likelihood()` call when there are insufficient data so
    that the test sampler can provide a detailed report to the user
    about the problem (recall that the typical lifetime of a cosmosis
    run involves running the test sampler first to weed out such
    problems, before a full run takes place).

    """

    def __init__(self, message, data):
        u"""Data to provide to the test sampler in case of likelihood data problems.

        The `message` is for the user generally; for use of `data` see
        the `test_sampler.execute()` method, currently at
        `samplers/test/test_sampler.py:29`.
        
        """
        super(MissingLikelihoodError, self).__init__(message)
        self.pipeline_data = data



class Pipeline(object):

    u"""A container of :class:`DataBlock`-processing :class:`Module`ʼs.

    Cosmosis consists of a large number of computational modules, which
    the user can arrange into pipelines in the configuration files
    before submitting a run.  This class encapsulates the basic concept
    of a pipeline, and provides methods for setting one up based on
    options and/or initial parameter vectors, and for running the
    pipeline to obtain a refinement of a :class:`DataBlock` (the
    prototypical use-case is Bayesian updating of prior likelihoods,
    though this is handled by the more specialized
    :class:`LikelihoodPipeline` defined below).

    Sometimes it will be possible to optimize execution of a pipeline by
    providing pre-emptive data which can be taken as the result of running
    the first few modules.  In this case a run will ‘bypass’ those modules
    and start the run at a `shortcut_module`, using the `shortcut_data` as
    the initial data set.

    """

    def __init__(self, arg=None, load=True):

        u"""Pipeline constructor.

        The (poorly named) `arg` needs to be some reference to a parameter 
        :class:`Inifile` which includes all the information to form this
        pipeline: it can be a list of filenames (.ini files and such) to
        read for the parameters, or a :class:`Inifile` object directly.

        If `load` is `True` then all the modules in the pipelineʼs
        configuration will be loaded into memory and initialized.

        """
        if arg is None:
            arg = list()

        if isinstance(arg, config.Inifile):
            self.options = arg
        else:
            self.options = config.Inifile(arg)

        #This will be set later
        self.root_directory = self.options.get("runtime", "root", fallback="cosmosis_none_signifier")
        if self.root_directory=="cosmosis_none_signifier":
            self.root_directory=None

        base_directory = self.base_directory()

        self.quiet = self.options.getboolean(PIPELINE_INI_SECTION, "quiet", fallback=True)
        self.debug = self.options.getboolean(PIPELINE_INI_SECTION, "debug", fallback=False)
        self.timing = self.options.getboolean(PIPELINE_INI_SECTION, "timing", fallback=False)
        shortcut = self.options.get(PIPELINE_INI_SECTION, "shortcut", fallback="")
        if shortcut=="": shortcut=None

        # initialize modules
        self.modules = []
        if load and PIPELINE_INI_SECTION in self.options.sections():
            module_list = self.options.get(PIPELINE_INI_SECTION,
                                           "modules", fallback="").split()

            self.modules = [
                module.Module.from_options(module_name,self.options,base_directory)
                for module_name in module_list
            ]

            self.shortcut_module=0
            self.shortcut_data=None
            if shortcut is not None:
                try:
                    index = module_list.index(shortcut)
                except ValueError:
                    raise ValueError("You tried to set a shortcut in "
                        "the pipeline but I do not know module %s"%shortcut)
                if index == 0:
                    print("You set a shortcut in the pipeline but it was the first module.")
                    print("It will make no difference.")
                self.shortcut_module = index



    def base_directory(self):
        u"""Return our `root_directory` according to the environment.

        Use the environment variable `COSMOSIS_SRC_DIR` if available,
        otherwise use this processʼs working directory.

        """
        if self.root_directory is None:
            try:
                self.root_directory = os.environ["COSMOSIS_SRC_DIR"]
                print("Root directory is ", self.root_directory)
            except KeyError:
                self.root_directory = os.getcwd()
                print("WARNING: Could not find environment variable")
                print("COSMOSIS_SRC_DIR. Module paths assumed to be relative")
                print("to current directory, ", self.root_directory)
        return self.root_directory



    def find_module_file(self, path):
        u"""Find a module file, which is assumed to be either absolute or relative to COSMOSIS_SRC_DIR"""
        return os.path.join(self.base_directory(), path)



    def setup(self):
        u"""Run all of our modulesʼ `setup()` routines."""
        
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
            global_sections = self.options.get("runtime", "global", fallback=" ")
            for global_section in global_sections.split():
                relevant_sections.append(global_section)

            config_block = config_to_block(relevant_sections, self.options)
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
        u"""Call every `module`ʼs `cleanup` method."""
        for module in self.modules:
            module.cleanup()



    def make_graph(self, data, filename):
        u"""Put a description of a graphical model in the graphviz format
        of a completed datablock `data' that was run on the pipeline, 
        in `filename`, illustrating the data flow of this pipeline.

        Graphviz tools can then be used to generate an actual image.

        The :class:`DataBlock` `data`ʼs attached log is used to determine
        the state of each module, and colourization is used in the graphic
        to indicate this.

        """
        try:
            import pygraphviz as pgv
        except ImportError:
            print("Cannot generate a graphical pipeline; please install the python package pygraphviz (e.g. with pip install pygraphviz)")
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
        for i in range(len(self.modules)-1):
            P.add_edge(norm_name(self.modules[i].name),norm_name(self.modules[i+1].name), color='lightskyblue', style='bold', arrowhead='none')
        # D = pydot.Cluster(label="Data", color='red', style='dashed')
        # G.add_subgraph(D)
        # #find
        log = [data.get_log_entry(i) for i in range(data.get_log_count())]
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
        u"""Run every module, in sequence, on DataBlock `data_package`.

        Apart from that the function goes to a lot of effort to provide
        run-time diagnostic information to the user.

        However, if any module returns anything but a zero status, the
        whole pipeline will cease to run and a :class:`ValueError` will be
        raised.

        Bear in mind the note on short-cuts in the class description
        above: if `shortcut_data` and `shortcut_module` are defined, then
        the pipeline will start at `shortcut_module` with `shortcut_data`
        as the initial parameter vector.

        """
        modules = self.modules
        first = (self.shortcut_data is None)
        if self.timing:
            start_time = time.time()
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

            if status is None:
                raise ValueError(("A module you ran, '{}', did not return a proper status value.\n"+
                    "It should return an integer, 0 if everything worked.\n"+
                    "Sorry to be picky but this kind of thing is important.").format(module))

            if self.debug:
                sys.stdout.write("Done %.20s status = %d \n" % (module,status))
                sys.stdout.flush()

            if self.timing:
                t2 = time.time()
                sys.stdout.write("%s took: %.3f seconds\n"% (module,t2-t1))

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
                print("Saving shortcut data")
                self.shortcut_data = data_package.clone()

        if self.timing:
            end_time = time.time()
            sys.stdout.write("Total pipeline time: {:.3} seconds\n".format(end_time-start_time))


        if not self.quiet:
            sys.stdout.write("Pipeline ran okay.\n")

        data_package.log_access("MODULE-START", "Results", "")
        # return something
        return True



class LikelihoodPipeline(Pipeline):

    u"""Very specialized pipeline designed specifically for the prototypical case of Bayes-computed posterior distributions.

    The point of a statistical updating pipeline is that the parameters in
    the datablocks passed down the pipe, as well as having currently
    estimated values, also have allowable ranges and possibly other
    constraints which the user may want to tinker before each run.  Thus
    there is a specialized layout of initialization files in the file
    system, and there is a modified expectation on the modules to perform
    simulation, compute the Bayesian evidence, hence log-likelihood.  The
    pipeline itself will aggregate the results and summarize the net
    effect of all the likelihood estimations, and thence compute the
    Bayesian posterior.

    Because of the necessity of working with distributions of values for
    each parameter, rather than just a scalar, the extra information is
    stored in a shadow array—another dictionary with the same keys but a
    complementary set of values to the original ones—of
    :class:`parameter`s to the :class:`datablock` which the base pipeline
    modifies (actually only a subset of them known as the `varied_params`:
    an array which references the interesting parameters in the full set).
    this shadow array (`parameters`) is often referred to simply as `p`,
    and the two arrays frequently need to be ‘zipped’ together and then
    ‘unzipped’ after computations have completed.

    """

    def __init__(self, arg=None, id="",override=None, load=True):
        u"""Construct a :class:`LikelihoodPipeline`.

        The arguments `arg` and `load` are used in the base-class
        initialization (see above).  The `id` is given to our `id_code`
        (which doesnʼt seem to have a purpose), and `override` is a
        dictionary of `(section, name)->value` which will override any
        settings for those parametersʼ values in the initialization files.
        
        """
        super(LikelihoodPipeline, self).__init__(arg=arg, load=load)

        if id:
            self.id_code = "[%s] " % str(id)
        else:
            self.id_code = ""
        self.n_iterations = 0

        values_file = self.options.get(PIPELINE_INI_SECTION, "values")
        self.values_filename=values_file
        priors_files = self.options.get(PIPELINE_INI_SECTION,
                                        "priors", fallback="").split()
        self.priors_files = priors_files

        self.parameters = parameter.Parameter.load_parameters(values_file,
                                                              priors_files,
                                                              override,
                                                              )

        self.reset_fixed_varied_parameters()

        self.print_priors()

        #We want to save some parameter results from the run for further output
        extra_saves = self.options.get(PIPELINE_INI_SECTION,
                                       "extra_output", fallback="")

        self.extra_saves = []
        for extra_save in extra_saves.split():
            section, name = extra_save.upper().split('/')
            self.extra_saves.append((section, name))

        self.number_extra = len(self.extra_saves)
        #pull out all the section names and likelihood names for later

        likelihood_names = self.options.get(PIPELINE_INI_SECTION,
                                            "likelihoods",
                                            fallback=NO_LIKELIHOOD_NAMES)
        if likelihood_names==NO_LIKELIHOOD_NAMES:
            self.likelihood_names = NO_LIKELIHOOD_NAMES
        else:
            self.likelihood_names = likelihood_names.split()

        # now that we've set up the pipeline properly, initialize modules
        self.setup()



    def print_priors(self):
        u"""Pretty-print a table of priors for human inspection."""
        print("")
        print("Parameter Priors")
        print("----------------")
        if self.parameters:
            n = max([len(p.section)+len(p.name)+2 for p in self.parameters])
        else:
            n=1
        for param in self.parameters:
            s = "{}--{}".format(param.section,param.name)
            print("{0:{1}}  ~ {2}" .format(s, n, param.prior))
        print("")

    def reset_fixed_varied_parameters(self):
        u"""Identify the sub-set of parameters which are fixed, and those which are to be varied."""
        self.varied_params = [param for param in self.parameters
                              if param.is_varied()]
        self.fixed_params = [param for param in self.parameters
                             if param.is_fixed()]



    def parameter_index(self, section, name):
        u"""Return the sequence number of the parameter `name` in `section`.

        If the parameter is not found then :class:`ValueError` will be raised.

        """
        i = self.parameters.index((section, name))
        if i==-1:
            raise ValueError("Could not find index of parameter %s in section %s"%(name, section))
        return i



    def set_varied(self, section, name, lower, upper):
        u"""Indicate that the parameter (`section`, `name`) is to be varied between the `lower` and `upper` bounds."""
        i = self.parameter_index(section, name)
        self.parameters[i].limits = (lower,upper)
        self.reset_fixed_varied_parameters()



    def set_fixed(self, section, name, value):
        u"""Indicate that the parameter (`section`, `name`) must be held fixed at `value`."""
        i = self.parameter_index(section, name)
        self.parameters[i].limits = (value, value)
        self.reset_fixed_varied_parameters()



    def output_names(self):
        u"""Return a list of strings, each the name of a non-fixed parameter."""
        param_names = [str(p) for p in self.varied_params]
        extra_names = ['%s--%s'%p for p in self.extra_saves]
        return param_names + extra_names



    def randomized_start(self):
        u"""Give each varied parameter an independent random value distributed according
        to the parameter prior.

        The return is a `NumPy` :class:`array` of the random values.

        """
        
        # should have different randomization strategies
        # (uniform, gaussian) possibly depending on prior?
        
        return np.array([p.random_point() for p in self.varied_params])



    def is_out_of_range(self, p):
        u"""Determine if any parameter is not in its allowed range."""
        return any([not param.in_range(x) for
                    param, x in zip(self.varied_params, p)])



    def denormalize_vector(self, p, raise_exception=True):
        u"""Convert an array of normalized parameter values, one for each varied parameter,
        in the range [0.0,1.0] into their original values using only the lower and upper limits of the parameter.
    
        Use denormalize_vector_from_prior to convert according to the prior instead.

        """
        return np.array([param.denormalize(x, raise_exception) for param, x
                         in zip(self.varied_params, p)])



    def denormalize_vector_from_prior(self, p):
        u"""Convert an array of normalized parameter values, one for each varied parameter,
        in the range [0.0,1.0] into their original values according to the prior for each parameter.

        i.e. 
        v -> x  such that \int_{-inf}^{x} p(x') dx' = v

        """
        return np.array([param.denormalize_from_prior(x) for param, x
                         in zip(self.varied_params, p)])



    def normalize_vector(self, p):
        u"""Convert an array of parameter values, one for each varied parameter,
         into a normalized form all in the range [0.0,1.0] using only the lower and upper limits for each parameter."""
        return np.array([param.normalize(x) for param, x
                         in zip(self.varied_params, p)])



    def normalize_matrix(self, c):
        u"""Roughly, return a correlation matrix corresponding to the 
        covariance matrix `c`, of `varied_params` values.

        Except that the elements of `c` are not probabilities but
        dimensional values, and the ‘normalization’ is relative to the
        range of values the ‘covariance’s can take given the lower and
        upper limits on the variates.

        """
        c = c.copy()
        n = c.shape[0]
        assert n==c.shape[1], "Cannot normalize a non-square matrix"
        for i in range(n):
            pi = self.varied_params[i]
            ri = pi.limits[1] - pi.limits[0]
            for j in range(n):
                pj = self.varied_params[j]
                rj = pj.limits[1] - pj.limits[0]
                c[i,j] /= (ri*rj)
        return c



    def denormalize_matrix(self, c, inverse=False):
        u"""Perform the inverse operation to the normalize_matrix function above.

        Note that if `inverse` is `True` the action is *exactly* the same
        as the function above, i.e. it *normalizes* the matrix.

        """
        c = c.copy()
        n = c.shape[0]
        assert n==c.shape[1], "Cannot normalize a non-square matrix"
        for i in range(n):
            pi = self.varied_params[i]
            ri = pi.limits[1] - pi.limits[0]
            for j in range(n):
                pj = self.varied_params[j]
                rj = pj.limits[1] - pj.limits[0]
                if inverse:
                    c[i,j] /= (ri*rj)
                else:
                    c[i,j] *= (ri*rj)
        return c



    def start_vector(self, all_params=False, as_array=True):
        u"""Return a vector of starting values for parameters.

        If `all_params` is specified as `True` then the return will include all
        our `parameters`, otherwise only the varying ones are included.

        If `as_array` is specified as `False` then a Python list is
        returned, otherwise, the default, a NumPy array is returned.

        """
        if all_params:
            p = [param.start for param in self.parameters]
        else:            
            p =[param.start for param in self.varied_params]
        if as_array:
            p = np.array(p)
        return p



    def min_vector(self, all_params=False):
        u"""Return a NumPy array of lower limits for the parameters in the pipeline.

        If `all_params` is specified as `True` then the return will 
        include all parameters, including fixed ones. Otherwise it will just be the 
        varying parameters.

        """
        if all_params:
            return np.array([param.limits[0] for
                 param in self.parameters])
        else:
            return np.array([param.limits[0] for
                         param in self.varied_params])



    def max_vector(self, all_params=False):
        u"""Return a NumPy array of upper limits for the parameters in the pipeline.

        If `all_params` is specified as `True` then the return will 
        include all parameters, including fixed ones. Otherwise it will just be the 
        varying parameters.

        """
        if all_params:
            return np.array([param.limits[1] for
                 param in self.parameters])
        else:
            return np.array([param.limits[1] for
                         param in self.varied_params])

    def build_starting_block(self, p, check_ranges=False, all_params=False):
        u"""Assemble :class:`DataBlock` data based on parameter values in `p`, and return it.


        If `check_ranges` is indicated, the function will return `None` if
        **any** of our parameters are out of their indicated range.

        If `all_params` is indicated, then the `p` run data will be
        assumed to match all the pipeline parameter, including fixed ones.
        Otherwise (the default) it should match the list ‘varied_params’, and all
        of our ‘fixed’ parameters are added to the run-set.

        """
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

        return data


    def run_parameters(self, p, check_ranges=False, all_params=False):
        u"""Assemble :class:`DataBlock` data based on parameter values in `p`, and run the pipeline on those data.

        If `check_ranges` is indicated, the function will return `None` if
        **any** of our parameters are out of their indicated range.

        If `all_params` is indicated, then the `p` run data will be
        assumed to match all the pipeline parameter, including fixed ones.
        Otherwise (the default) it should match the list ‘varied_params’, and all
        of our ‘fixed’ parameters are added to the run-set.

        """
        data = self.build_starting_block(p, check_ranges=check_ranges, all_params=all_params)

        if self.run(data):
            return data
        else:
            sys.stderr.write("Pipeline failed on these parameters: {}\n".format(p))
            return None



    def create_ini(self, p, filename):
        u"""Dump the parameters `p` as a new ini file at `filename`"""
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



    def prior(self, p, all_params=False, total_only=True):
        u"""Compute the probability of all values in `p` based on their prior distributions.

        The array `p` should match the length of of all of our `parameters` if
        `all_params` is `True`, and our `varied_params` otherwise.

        If `total_only` is `True` (the default), then the scalar sum of
        all the prior probabilities is returned.  Otherwise a list
        of pairs is returned, with each element a stringified version of
        the parameter name, and the prior probability:
        [(name1, prior1), (name2,prior2), ...]

        """
        if all_params:
            params = self.parameters
        else:
            params = self.varied_params
        priors = [(str(param),param.evaluate_prior(x)) for param,x in zip(params,p)]
        if total_only:
            return sum(pr[1] for pr in priors)
        else:
            return priors



    def posterior(self, p, return_data=False, all_params=False):
        u"""Use the above methods to obtain prior and updated log-likelihoods, sum together to get Bayesian posterior.

        The argument `p` is the set of :class:`Parameter`s which shadows
        `self.varied_params`, unless `all_params` is specified as `True`
        in which case it shadows `self.parameters`.

        The method returns two or three values depending on `return_data`:

        * The posterior;

        * a vector (NumPy array) of updated parameter values as specified
          in `self.extra_saves`;

        * if `return_data` was specified, the updated data block.

        If there is a problem anywhere in the computations which does
        *not* cause a run-time exception to be raised—including the case
        where a parameter goes outside of its alloted range—, then
        `-numpy.inf` will be returned as the final posterior (i.e., zero
        probability of this set of parameter values being correct).

        """

        priors = self.prior(p, all_params=all_params, total_only=False)
        # The total prior
        prior = sum(pr[1] for pr in priors)
        if prior == -np.inf:
            if not self.quiet:
                sys.stdout.write("Proposed outside bounds\nPrior -infinity\n")
            if return_data:
                return prior, np.repeat(np.nan, self.number_extra), None
            return prior, np.repeat(np.nan, self.number_extra)

        try:
            results = self.likelihood(p, return_data=return_data, all_params=all_params)
            error = False

            like = results[0]
            extra = results[1]
            
            if return_data:
                data = results[2]
                for name,pr in priors:
                    data["priors", name] = pr

        except Exception:
            error = True
            # If we are 
            if self.debug:
                sys.stderr.write("\n\nERROR: there was an exception running the likelihood:\n")
                sys.stderr.write("\n\Because you have debug=T I will let this kill the chain.\n")
                sys.stderr.write("The input parameters were:{}\n".format(repr(p)))
                raise

            sys.stderr.write("\n\nERROR: there was an exception running the likelihood:\n")
            sys.stderr.write("The input parameters were:{}\n".format(repr(p)))
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("You should fix this but for now I will return NaN for the likelihood (because you have debug=F)\n\n")

            # Replace with bad values
            like = -np.inf
            data = None
            extra = np.repeat(np.nan, self.number_extra)

        if return_data:
            return prior + like, extra, data
        else:
            return prior + like, extra

    def _set_likelihood_names_from_block(self, data):
        likelihood_names = []
        for _,key in data.keys(cosmosis_py.section_names.likelihoods):
            if key.endswith("_like"):
                name = key[:-5]
                likelihood_names.append(name)
        self.likelihood_names = likelihood_names

    def _extract_likelihoods(self, data):
        "Extract the likelihoods from the block"

        section_name = cosmosis_py.section_names.likelihoods

        # First run.  If we have not set the likelihood names in the parameter
        # file then get them from the block
        if self.likelihood_names == NO_LIKELIHOOD_NAMES:
            self._set_likelihood_names_from_block(data)

            if not self.quiet:
                # Tell the suer what we found
                print("Likelihoods not set in parameter file, so checking what is generated:")
                for name in self.likelihood_names:
                    print("Found likelihood named {}".format(name))
                if not self.likelihood_names:
                    print("No likelihoods found")

        # loop through named likelihoods and sum their values
        likelihoods = []
        for likelihood_name in self.likelihood_names:
            try:
                L = data.get_double(section_name,likelihood_name+"_like")
                likelihoods.append(L)
                if not self.quiet:
                    print("    Likelihood {} = {}".format(likelihood_name, L))
            # Complain if not found
            except block.BlockError:
                raise MissingLikelihoodError(likelihood_name, data)

        # Total likelihood
        like = sum(likelihoods)

        # DM: Issue #181: Zuntz: replace NaN's with -inf's in posteriors and
        #                 likelihoods.
        if np.isnan(like):
            like = -np.inf

        if not self.quiet and self.likelihood_names:
            sys.stdout.write("Likelihood total = {}\n".format(like))

        return like

        
    def likelihood(self, p, return_data=False, all_params=False):
        u"""Run the simulation pipeline, computing any log-likelihoods in the pipeline 
        given the given input parameter values, and return the sum of these.

        The parameter vector `p` must match the length of `self.varied_params`, unless
        `all_params` is specified as `True` in which case it must match
        `self.parameters', i.e. must correspond to the complete parameter
        set.

        If `return_data` are requested, then the updated data block will
        be returned as the third return item.

        The return will consist of two or three items, depending on
        `return_data`:

          * A scalar holding the sum of all computed log-likelihoods of
            the updated parameter value vector;

          * a vector (NumPy array) of updated parameter values as
            specified in `self.extra_saves`;

          * if `return_data` was specified, the updated data block.

        If anything goes wrong in any of the computation which does *not*
        result in a run-time error being raised (which would include the
        case of a parameter going outside of its stipulated limits), then
        the returned log-likelihood will be `-np.inf`.

        """

        data = self.run_parameters(p, all_params=all_params)
        if data is None:
            if return_data:
                return -np.inf, np.repeat(np.nan, self.number_extra), data
            else:
                return -np.inf, np.repeat(np.nan, self.number_extra)

        like = self._extract_likelihoods(data)

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



def config_to_block(relevant_sections, options):
    u"""Compose :class:`DataBlock` of parameters only in `relevant_sections` of complete set of `options`."""
    config_block = block.DataBlock()

    for (section, name), value in options:
        if section in relevant_sections:
            # add back a default section?
            val = options.gettyped(section, name)
            if val is not None:
                config_block.put(section, name, val)
    return config_block
