from .output_base import OutputBase
from . import utils
import numpy as np
import os
from glob import glob
from collections import OrderedDict

comment_indicator = "_cosmosis_comment_indicator_"


class TextColumnOutput(OutputBase):
    FILE_EXTENSION = ".txt"
    _aliases = ["text", "txt"]

    def __init__(self, filename, rank=0, nchain=1, delimiter='\t'):
        super(TextColumnOutput, self).__init__()
        self.delimiter = delimiter

        #If filename already ends in .txt then remove it for a moment
        if filename.endswith(self.FILE_EXTENSION):
            filename = filename[:-len(self.FILE_EXTENSION)]

        if nchain > 1:
            self._filename = "%s_%d%s" % (filename, rank+1, 
                                          self.FILE_EXTENSION)
        else:
            self._filename = filename + self.FILE_EXTENSION

        self._file = open(self._filename, "w")

        #also used to store comments:
        self._metadata = OrderedDict()

    def _close(self):
        self._file.close()

    def _begun_sampling(self, params):
        #write the name line
        name_line = '#'+self.delimiter.join(c[0] for c in self.columns) + '\n'
        self._file.write(name_line)
        #now write any metadata.
        #text mode does not support comments
        for (key,(value,comment)) in self._metadata.items():
            if key.startswith(comment_indicator):
                self._file.write("## %s\n"%value.strip())
            elif comment:
                self._file.write('#{k}={v} #{c}\n'.format(k=key,v=value,c=comment))
            else:
                self._file.write('#{k}={v}\n'.format(k=key,v=value,c=comment))
        self._metadata={}

    def _write_metadata(self, key, value, comment=''):
        #We save the metadata until we get the first 
        #parameters since up till then the columns can
        #be changed
        #In the text mode we cannot write more metadata
        #after sampling has begun (because it goes at the top).
        #What should we do?
        self._metadata[key]= (value, comment)

    def _write_comment(self, comment):
        #save comments along with the metadata - nice as 
        #preserves order
        self._metadata[comment_indicator +
                       "_%d" % (len(self._metadata))] = (comment,None)

    def _write_parameters(self, params):
        line = self.delimiter.join(str(x) for x in params) + '\n'
        self._file.write(line)

    def _write_final(self, key, value, comment=''):
        #I suppose we can put this at the end - why not?
        c=''
        if comment:
            c='  #'+comment
        self._file.write('#{k}={v}{c}\n'.format(k=key,v=value,c=c))

    @classmethod
    def from_options(cls, options):
        #look something up required parameters in the ini file.
        #how this looks will depend on the ini 
        filename = options['filename']
        delimiter = options.get('delimiter', '\t')
        rank = options.get('rank', 0)
        nchain = options.get('parallel', 1)
        return cls(filename, rank, nchain, delimiter=delimiter)

    @classmethod
    def load_from_options(cls, options):
        filename = options['filename']
        delimiter = options.get('delimiter', '\t')

        cut = False
        if filename.endswith(cls.FILE_EXTENSION):
            filename = filename[:-len(cls.FILE_EXTENSION)]
            cut = True
        # first look for serial file
        if os.path.exists(filename+cls.FILE_EXTENSION):
            datafiles = [filename+cls.FILE_EXTENSION]
        elif os.path.exists(filename) and not cut:
            datafiles = [filename]
        else:
            datafiles = glob(filename+"_[0-9]*"+cls.FILE_EXTENSION)
            if not datafiles:
                raise RuntimeError("No datafiles found!")

        #Read the metadata
        started_data = False
        metadata = []
        final_metadata = []
        data = []
        comments = []

        for datafile in datafiles:
            print 'LOADING CHAIN FROM FILE: ', datafile
            chain = []
            chain_metadata = {}
            chain_final_metadata = {}
            chain_comments = []
            for i,line in enumerate(open(datafile)):
                line = line.strip()
                if not line: continue
                if line.startswith('#'):
                    #remove the first #
                    #if there is another then this is a comment,
                    #not metadata
                    line=line[1:]
                    if i == 0:
                        column_names = line.split()
                    elif line.startswith('#'):
                        comment = line[1:]
                        chain_comments.append(comment)
                    else:
                        #parse form '#key=value #comment'
                        if line.count('#') == 0:
                            key_val = line.strip()
                            comment = ''
                        else:
                            key_val, comment = line.split('#', 1)
                        key,val = key_val.split('=',1)
                        val = utils.parse_value(val)
                        if started_data:
                            chain_final_metadata[key] = val
                        else:
                            chain_metadata[key] = val
                else:
                    started_data = True
                    words = line.split(delimiter)
                    vals = [float(word) for word in words]
                    chain.append(vals)
            
            data.append(np.array(chain))
            metadata.append(chain_metadata)
            final_metadata.append(final_metadata)
            chain_comments.append(comments)
        return column_names, data, metadata, comments, final_metadata
