from __future__ import print_function
from builtins import zip
from builtins import map
import itertools
import collections
import numpy as np

from .. import ParallelSampler


INI_SECTION = "importance"


def task(p):
    r = importance_pipeline.posterior(p)
    return r


class ImportanceSampler(ParallelSampler):
    #there are a bunch of these really, but we 
    #need to add them manually so that we can
    #keep the likelihood column at the end
    #so the postprocessors don't need to be re-written
    sampler_outputs = []
    parallel_output = False

    def config(self):
        global importance_pipeline
        importance_pipeline = self.pipeline
        self.input_filename = self.ini.get(INI_SECTION, "input")
        self.nstep = self.ini.getint(INI_SECTION, "nstep", fallback=128)
        self.add_to_likelihood = self.ini.getboolean(INI_SECTION, "add_to_likelihood", fallback=False)
        self.converged = False
        if self.is_master():
            self.load_samples(self.input_filename)

    def load_samples(self, filename):
        options = {"filename":filename}
        col_names, cols, metadata, comments, final_metadata = self.output.__class__.load_from_options(options)
        # pull out the "post" column first
        col_names = [name.lower() for name in col_names]
        likelihood_index = col_names.index('post')
        if likelihood_index<0:
            raise ValueError("I could not find a 'post' column in the chain %s"%filename)
        self.original_likelihoods = cols[0].T[likelihood_index]

        #We split the parameters into three groups:
        #   - ones that we have listed as varying
        #   - one that are fixed - WHAT SHOULD WE DO ABOUT THESE???
        #   - extras ones - these should be saved and put in the output

        self.original_extras = collections.OrderedDict()
        self.samples = []
        self.varied_params = []
        self.number_samples = len(cols[0])
        print("Have %d samples from old chain." % self.number_samples)
        for code,col in zip(col_names, cols[0].T):
            #we have already handled the likelihood
            if code=='post':continue
            #parse the header names in to (section,name)
            bits = code.split("--")
            if len(bits)==2:
                section, name = bits
                #No parameter should be varied in the old chain
                #but listed as fixed here
                if (section,name) in self.pipeline.fixed_params:
                    print("WARNING: %s varied in old chain now fixed.  I will fix it <--------- Read this warning." % name)
                elif (section,name) in self.pipeline.varied_params:
                    #Record the values of this parameter for later importance
                    #sampling
                    print("Found column in both pipelines:", code)
                    self.samples.append(col)
                    self.varied_params.append(bits)
                else:
                    print("Found column just in old pipeline:", code)
                    #This parameter was varied in the old code but is not
                    #here.  So we just save it for output
                    self.original_extras[code] = col
                    self.output.add_column(code, float)
            # anything here must be a sampler-specific 
            else:
                print("Found non-parameter column:", code)
                self.original_extras[code] = col
                if code=="weight":
                    code="old_weight"
                    print("Renaming weight -> old_weight")
                self.output.add_column(code, float)

        #Now finally add our actual two sampler outputs, old_like and like
        #P is the original likelihood in the old chain we have samples from
        #P' is the new likelihood.
        self.output.add_column("old_post", float) #This is the old likelihood, log(P)
        self.output.add_column("log_weight", float) #This is the log-weight, the ratio of the likelihoods
        self.output.add_column("post", float) #This is the new likelihood, log(P')


        #Now we need to reorder the list to match the order our pipeline is expecting
        #If a parameter is not listed we use the starting value
        reordered_cols = []
        for p in self.pipeline.varied_params:
            try:
                i = self.varied_params.index(p)
            except ValueError:
                i=-1
            if i>=0:
                col = self.samples[i]
            else:
                col = np.repeat(p.start,self.number_samples)
            reordered_cols.append(col)
        self.samples = np.transpose(reordered_cols)

        self.current_index = 0

    def execute(self):
        self.output.comment("Importance sampling from %s"%self.input_filename)

        #Pick out a chunk of samples to run on
        start = self.current_index
        end = start+self.nstep
        samples_chunk = self.samples[start:end]

        #Run the pipeline on each of the samples
        if self.pool:
            results = self.pool.map(task, samples_chunk)
        else:
            results = list(map(task, samples_chunk))

        #Collect together and output the results
        for i,(sample, (new_like, extra)) in enumerate(zip(samples_chunk, results)):
            #We already (may) have some extra values from the pipeline
            #as derived parameters.  Add to those any parameters used in the
            #old pipeline but not the new one
            extra = list(extra)
            for col in list(self.original_extras.values()):
                extra.append(col[start+i])
            #and then the old and new likelihoods
            old_like = self.original_likelihoods[start+i]
            if self.add_to_likelihood:
                new_like += old_like
            weight = new_like-old_like
            extra = list(extra) + [old_like,weight,new_like]
            #and save results
            self.output.parameters(sample, extra)
        #Update the current index
        self.current_index+=self.nstep

    def is_converged(self):
        return self.current_index>=self.number_samples
