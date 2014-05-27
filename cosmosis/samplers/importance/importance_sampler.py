import itertools
import numpy as np

from .. import ParallelSampler


INI_SECTION = "importance"


def task(p):
    return importance_pipeline.posterior(p)


class ImportanceSampler(ParallelSampler):

    def config(self):
        global importance_pipeline
        importance_pipeline = self.pipeline
        self.input_filename = self.ini.get(INI_SECTION, "input-file")
        self.nstep = self.ini.getint(INI_SECTION, "nstep", 128)
        self.converged = False
        if self.is_master():
            self.load_samples(self.input_filename)

    def load_samples(self, filename):
        col_names, cols, metadata, final_metadata = self.output.__class__.load(filename)
        # pull out the "like" column first
        col_names = [name.lower() for name in col_names]
        likelihood_index = col_names.index('like')
        if likelihood_index<0:
            raise ValueError("I could not find a 'like' column in the chain %s"%filename)
        self.original_likelihoods = cols[likelihood_index]

        #We split the parameters into three groups:
        #   - ones that we have listed as varying
        #   - one that are fixed - WHAT SHOULD WE DO ABOUT THESE???
        #   - extras ones - these should be saved and put in the output

        self.original_extras = {}
        self.samples = []
        self.varied_params = []
        self.number_samples = len(cols[0])
        for code,col in zip(col_names, cols):
            #we have already handled the likelihood
            if code=='like':continue
            #parse the header names in to (section,name)
            bits = code.split("--")
            if len(bits)==2:
                section, name = bits
                #No parameter should be varied in the old chain
                #but listed as fixed here
                if (section,name) in self.pipeline.fixed_params:
                    raise ValueError("A parameter %s varied in the old chain "
                        "is listed as fixed in the new one"%code)
                if (section,name) in self.pipeline.varied_params:
                    #Record the values of this parameter for later importance
                    #sampling
                    self.samples.append(col)
                    self.varied_params.append(bits)
                else:
                    #This parameter was varied in the old code but is not
                    #here.  So we just save it for output
                    self.original_extras[code] = col
            # anything here must be a sampler-specific 
            else:
                self.original_extras[code] = col

        #Now we need to reorder the list to match the order our pipeline is expecting
        #If a parameter is not listed we use the starting value
        reordered_cols = []
        for p in self.pipeline.varied_params:
            #This is the index of this parameter in the old chain
            i = self.varied_params.index(p)
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
            results = map(task, samples_chunk)

        for i,(sample, (new_like, extra)) in enumerate(itertools.izip(samples_chunk, results)):
            old_like = self.original_likelihoods[start+i]
            importance = new_like - old_like
            #print sample, new_like, extra
            extra['importance'] = importance
            extra['old_like'] = old_like
            self.output.parameters(sample, extra)
        #Update the current index
        self.current_index+=self.nstep

    def is_converged(self):
        return self.current_index>=self.number_samples
