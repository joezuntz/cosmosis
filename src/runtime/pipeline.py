import os
import ctypes
import sys
import string
import numpy as np
import ConfigParser
import time

import utils
import config 
import section_names
import parameter
import prior
import module

PIPELINE_INI_SECTION = "pipeline"

class Pipeline(object):
    def __init__(self, arg=None, quiet=False, debug=False, timing=False):
        """Initialize with a single filename or a list of them, a ConfigParser, or nothing for an empty pipeline"""
        if arg is None: arg = list()
        if isinstance(arg, ConfigParser.ConfigParser):
            self.options = arg
        else:
            self.options = inifiles.Inifile.from_file(arg)

        self.quiet=quiet
        self.debug=debug
        self.timing=timing

        # initialize modules
        self.modules = []
        if PIPELINE_INI_SECTION in self.options.sections():
            rootpath = self.get_option(PIPELINE_INI_SECTION,"root",os.curdir)

            for module_name in self.get_option( PIPELINE_INI_SECTION, "module", "").split():
                # identify module file
                filename = self.get_option(module_name, "file")

                # identify relevant functions
                setup_function = self.get_option(module_name, "function", "setup")
                exec_function = self.get_option(module_name, "setup", "execute")
                cleanup_function = self.get_option(module_name, "cleanup", "cleanup")

                modules.append( module.Module(module_name, 
                                              filename, 
                                              setup_function,
                                              exec_function,
                                              cleanup_function,
                                              rootpath) )

            # setup modules after ensuring they all exist and are loadable
            self.setup()

    def setup(self):
        if self.timing:
            timings = [time.clock()]

        for module in self.modules: 
            # identify parameters needed for module setup
            relevant_sections = [PIPELINE_INI_SECTION, "general", "logging", "debug", module_name]
            relevant_options = []
            for section, param, value in self.options:
                if section in relevant_sections:
                    if section == module_name:
                        relevant_options.append((options.default_section, param, value))
                    relevant_options.append((section, param, value))
            #relevant_options.append((options.DesOptionPackage._default_section, "module_name", m

            module.setup()

            if self.timing:
                timings.append(time.clock())

        if not self.quiet:
            sys.stdout.write("Setup all pipeline modules")

        if self.timing:
            timings.append(time.clock())
            sys.stdout.write("Module timing:\n")
            for name, t2,t1 in zip(self.modules, timings[1:], timings[:-1]):
                sys.stdout.write("%s %f\n"%(name, t2-t1))

    def cleanup(self):
        for module in self.modules:
            module.cleanup()

    def run(self, package):
        if self.timing:
            timings = [time.clock()]

        for module in self.modules:
            if self.debug:
                sys.stdout.write("Running %.20s ... " % module)
                sys.stdout.flush()
        
            status = module.execute( package )

            if self.timing:
                timings.append(time.clock())

            if status:
                if not self.quiet:
                    sys.stderr.write("Error running pipeline (%d)- hopefully printed above here.\n")
                    sys.stderr.write("Aborting this run and returning error status.\n")

                    if self.timing:
                        sys.stdout.write("Module timing:\n")
                        for name,t2,t1 in zip(self.module_names[:], timings[1:], timings[:-1]):
                            sys.stdout.write("%s %f\n"%(name, t2-t1))

        if not self.quiet:
            sys.stdout.write("Pipeline ran okay.")

        if self.timing:
            sys.stdout.write("Module timing:\n")
            for name, t2,t1 in zip(self.modules, timings[1:], timings[:-1]):
                sys.stdout.write("%s %f\n"%(name, t2-t1))

        # return something

    def get_option(self,section,name,default=None):
        try:
            return self.options.get(section,name)
        except ConfigParser.NoOptionError:
            if default is not None:
                return default
            raise ValueError("Could not find entry in the ini file for section %s, parameter %s, which was needed." % (section,name))


class LikelihoodPipeline(Pipeline):
    def __init__(self, arg=None, id="", debug=False, quiet=False, timing=False):
        super(LikelihoodPipeline, self).__init__(arg=arg, quiet=quiet, debug=debug, timing=timing)

        values_file = self.get_option(PIPELINE_INI_SECTION, "values")
        priors_files = self.get_option(PIPELINE_INI_SECTION, "priors", "")

        self.parameters = parameter.Parameter.load_parameters(values_file, priors_files)

        if id:
            self.id_code = "[%s] " % str(id)
        else:
            self.id_code = ""
        self.n_iterations = 0

        #We want to save some parameter results from the run for further output
        #extra_saves = self.get_option(PIPELINE_INI_SECTION,"extra_output", "")
        #self.extra_saves = []
        #for extra_save in extra_saves.split():
        #    section, name = extra_save.upper().split('/')
        #    section = getattr(section_names.section_names, section.lower(), section)
        #
        #    if not (section,name) in parameters:
        #        raise 
        #    self.extra_saves.append((section, name))


        #pull out all the section names and likelihood names for later
        #self.section_names = set([name for (name, param, value) in self.fixed_params+self.varied_params])
        self.likelihood_names = self.get_option(PIPELINE_INI_SECTION, "likelihoods").split()

    def randomized_start(self):
        return np.array([p.random_point() for p in self.varied_parameters])

    def is_out_of_range(self, params_by_section):
        out_of_range = []
        for (section, param, param_range) in self.varied_params:
            value = params_by_section[section][param]
            (pmin, pstart, pmax) = param_range
            if value<pmin:
                out_of_range.append("%s--%s=%.4g<%.4g"%(section,param,value,pmin))
            elif value>pmax:
                out_of_range.append("%s--%s=%.4g>%.4g"%(section,param,value,pmax))
        if out_of_range:
            if not self.quiet:
                print "Params out of range: " +  ('  '.join(out_of_range))
                print "Zero likelihood"
            return True
        return False            


    def set_run_parameters(self, p):
        #Set up the sections of parameters
        params_by_section = { section:{} for section in self.section_names }

        #Loop through the fixed parameters setting their values
        help_notes = []
        for (section, param, param_start) in self.fixed_params:
            params_by_section[section][param] = param_start

        #Now loop the varied ones.
        for (section, param, param_range), value in zip(self.varied_params, p):
            params_by_section[section][param] = value
            if not self.quiet: help_notes.append("%s--%s=%e" % (section,param,value))

        #Check for parameters being above the min/max values they are allowed.
        if self.is_out_of_range(params_by_section):
            return None
        
        if not self.quiet:
            help_text = '  '.join(help_notes)
            print "%sRunning pipeline iteration %d: %s" % (self.id_code, self.n_iterations, help_text)
        return params_by_section

#   def write_values_file(self, p, output_file):
#       """ Turn a parameter vector into a values file """
#       quiet = self.quiet
#       self.quiet=False
#       params_by_section = self.set_run_parameters(p)
#       self.quiet=quiet
#       if params_by_section is None:
#           print "Bad parameters:"
#           print p
#           return  
#       if isinstance(output_file,file):
#           f = output_file
#       else:
#           f = open(output_file, 'w')
#       for section_name, section_dict in params_by_section.items():
#           section_name = utils.section_friendly_names[section_name]
#           f.write('[%s]\n'%section_name)
#           for name, value in section_dict.items():
#               f.write("%-18s = %s\n" % (name,str(value)))
#           f.write("\n")
#       if not isinstance(output_file,file):
#           f.close()

#   @staticmethod
#   def data_package_from_start(start=None):
#       if isinstance(start, data_package.DesDataPackage):
#           data = start
#       elif isinstance(start, basestring):
#           data = data_package.DesDataPackage.from_file(start)
#       elif start is None:
#           data = data_package.DesDataPackage()
#       else:
#           raise ValueError("Could not convert object %r into a data package (internal error; report this please)"%start)
#       return data

    def run_vector(self,p,start=None):
        params_by_seution = self.set_run_parameters(p)
        if params_by_section is None:
            return None
        return self.run_parameters(params_by_section, start=start)

#   def starting_vector(self):
#       return np.array([r[2][1] for r in self.varied_params])

    def run_parameters(self, params_by_section, start=None, check_ranges=False):
        if params_by_section is None:
            return None
        if check_ranges:
            if self.is_out_of_range(params_by_section):
                return None

        
        data = self.data_package_from_start(start)
        data.add_mixed_params(params_by_section)
        data = self.run(data)
        return data

    def run_ranges_file(self, ranges_file, start=None):
        params_by_section = ranges_file.to_fixed_parameter_dicts()
        return self.run_parameters(params_by_section, start=start)


    def extract_likelihood(self, data):
        like = 0.0
        like_section = section_names.likelihoods
        for likelihood_name in self.likelihood_names:
            like += data.get_param(like_section,likelihood_name+"_LIKE")
        return like

#   def header_text(self):
#       param_names = ["%s--%s"%(p[0],p[1]) for p in self.varied_params]
#       output = ['# Likelihood   ']
#       output+=["%-20s"%p for p in param_names]
#       output+= ['%s--%s'%param for param in self.extra_saves]
#       return '\t'.join(output)

#   def parameter_output_text(self, like, params, extra_data):
#       output = ["%-20s"%str(like)]
#       for (section,name,_) in self.varied_params:
#           value = params[section][name]
#           output.append("%-20s"%str(value))
#       for (section,name,_) in self.extra_saves:
#           value = extra_data[section][name]
#           output.append("%-20s"%str(value))
#       return '\t'.join(output)

    def denormalize_vector(self, p):
        p_out = []
        for (section, param, param_range), value in zip(self.varied_params, p):
            param_min, param_start, param_max = param_range
            p_i = value*(param_max-param_min) + param_min
            p_out.append(p_i)
        return np.array(p_out)

    def normalize_vector(self, p):
        p_out = []
        for (section, param, param_range), value in zip(self.varied_params, p):
            param_min, param_start, param_max = param_range
            p_i = (value - param_min)/(param_max - param_min)
            p_out.append(p_i)
        return np.array(p_out)

#   @staticmethod
#   def nested_dict_to_single_dict(D):
#       output = {}
#       for section, d in D.items():
#           for name,value in d.items():
#               output[(section,name)] = value
#       return output

    def prior(self, p):
        if not self.priors_calculator:
            return 0.0
        if isinstance(p, dict):
            return self.priors_calculator.get_prior_for_nested_dict(p)
        else:           
            param_dict = {}
            for (section, param, param_range), value in zip(self.varied_params, p):
                param_dict[(section,param)] = value
            return self.priors_calculator.get_prior_for_collected_parameter_dict(param_dict)

    def posterior(self, p, filename=None):
        prior = self.prior(p)
        if prior == -np.inf:


            return prior, utils.everythingIsNan
        like, extra = self.likelihood(p, filename=filename)
        return prior + like, extra

    def likelihood(self, p):
        #Set the parameters by name from the parameter vector
        #If one is out of range then return -infinity as the log-likelihood
        #i.e. likelihood is zero.  Or if something else goes wrong do the same
        if isinstance(p, dict):
            data = self.run_parameters(p)
        else:
            data = self.run_vector(p)

        if data is None:
            return -np.inf, utils.everythingIsNan

        # Otherwise we loop through the named likelihoods
        # and sum all the values we find for them
        like = self.extract_likelihood(data)
        if not self.quiet: 
            sys.stdout.write("Likelihood %e"% (like,))
        
        extra_saves = {}
        for (section, name) in self.extra_saves:
            value = data.get_param(section, name, np.nan)
            extra_saves['%s--%s'%(section,name)] = value
        extra_saves['LIKE'] = like

        self.n_iterations += 1
        return like, extra_saves
