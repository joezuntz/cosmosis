from __future__ import with_statement


from pymc.database import base, ram
import pymc
import os, datetime, shutil, re
import numpy as np
from numpy import array
import string

__all__ = ['Trace', 'Database']

CHAIN_NAME = 'Chain_%d'

class Trace(ram.Trace):
    def tally(self, chain):
        super(Trace, self).tally(chain)
        self.db.alert_tallied(self, self._getfunc())
    pass

class Database(base.Database):
    """Txt Database class."""

    def __init__(self, dbname=None):
        """Create a Txt Database.

        :Parameters:
        dbname : string
          Name of the directory where the traces are stored.
        dbmode : {a, r, w}
          Opening mode: a:append, w:write, r:read.
        """
        self.__name__ = 'textfile'
        self._filename = dbname
        self.__Trace__ = Trace

        self.trace_names = []   # A list of sequences of names of the objects to tally.
        self._ordered_trace_names = []   # A list of sequences of names of the objects to tally.
        self._traces = {} # A dictionary of the Trace objects.
        self.chains = 0

        self._output_file = open(self._filename,'w')
        self._output_node_dictionary = {}

        self._tallied_traces = {}

    def alert_tallied(self, trace, value):
        self._tallied_traces[trace] = value
        if len(self._tallied_traces) == len(self._traces):
#            print self._traces.keys()
#            for name, trace in self._traces.items():
            for name in self._ordered_trace_names:
                trace = self._traces[name]
                value = self._tallied_traces[trace]
                if isinstance(value,np.ndarray) and value.shape==(1,):
                    value = float(value)
                if name=='deviance': 
                    value = -value/2.0
                else:
                    trace_node = self._output_node_dictionary[name]
                    xmin = trace_node._original_lower
                    xmax = trace_node._original_upper
                    value = xmin + value * (xmax-xmin)
                self._output_file.write('  %s' % str(value))
            self._output_file.write('\n')
            self._output_file.flush()
            self._tallied_traces={}
    def _initialize(self, chain, length):
        super(Database, self)._initialize(chain, length)
        self._output_file.write('#  ')
        keys = self._traces.keys()
        keys.sort()
        keys.remove("deviance")
        keys  = ['deviance'] + keys
        self._ordered_trace_names = keys[:]
        for name in keys:
            if name=='deviance':
                name='likelihood'
            else:
                #something else to do - set name
                for node in self.model.nodes:
                    if node.__name__==name:
                        trace_node=node
                        break
                else:
                    raise ValueError("Internal error in db - JZ's fault")
                self._output_node_dictionary[name] = trace_node
            # self._ordered_trace_names.append(name)
            self._output_file.write('%s  ' % name)
        self._output_file.write('\n')



    def savestate(self, state):
        """Save the sampler's state in a state.txt file."""
        oldstate = np.get_printoptions()
        np.set_printoptions(threshold=1e6)
        try:
            filename = self._filename + ".state"
            with open(filename, 'w') as f:
                f.write(str(state))
        finally:
            np.set_printoptions(**oldstate)


    def _get_rescaled_array(self,trace,name,chain=-1):
            arr = trace.gettrace(chain=chain)
            arr = np.array(arr)
            if name=='deviance':
                name = 'likelihood'
                return -0.5*arr

            for node in self.model.nodes:
                if node.__name__==name:
                    trace_node=node
                    break
            else:
                raise ValueError("Internal error in db - JZ's fault")
            xmin = trace_node._original_lower
            xmax = trace_node._original_upper
            return xmin + arr*(xmax-xmin)


    def _finalize(self, chain=-1):
        super(Database,self)._finalize(chain)
        """Finalize the chain for all tallyable objects."""
        # chain = range(self.chains)[chain]
        # arrays = []
        # names = []
        # for name, trace in self._traces.items():
        #     arr = self._get_rescaled_array(trace,name,chain)
        #     names.append(name)
        #     arrays.append(arr)
        #     trace._finalize(chain)
        # arrays = np.array(arrays).T
        # with open(self._filename,'w') as f:
        #     f.write('# ')
        #     f.write('  '.join(names))
        #     f.write('\n')
        #     np.savetxt(f, arrays, fmt='%.5e')
        self._output_file.close()
        self.commit()



# def load(dirname):
#     """Create a Database instance from the data stored in the directory."""
#     if not os.path.exists(dirname):
#         raise AttributeError('No txt database named %s'%dirname)

#     db = Database(dirname, dbmode='a')
#     chain_folders = [os.path.join(dirname, c) for c in db.get_chains()]
#     db.chains = len(chain_folders)

#     data = {}
#     for chain, folder in enumerate(chain_folders):
#         files = os.listdir(folder)
#         funnames = funname(files)
#         db.trace_names.append(funnames)
#         for file in files:
#             name = funname(file)
#             if name not in data:
#                 data[name] = {} # This could be simplified using "collections.defaultdict(dict)". New in Python 2.5
#             # Read the shape information
#             with open(os.path.join(folder, file)) as f:
#                 f.readline(); shape = eval(f.readline()[16:])
#                 data[name][chain] = np.loadtxt(os.path.join(folder, file), delimiter=',').reshape(shape)
#                 f.close()


#     # Create the Traces.
#     for name, values in six.iteritems(data):
#         db._traces[name] = Trace(name=name, value=values, db=db)
#         setattr(db, name, db._traces[name])

#     # Load the state.
#     statefile = os.path.join(dirname, 'state.txt')
#     if os.path.exists(statefile):
#         with open(statefile, 'r') as f:
#             db._state_ = eval(f.read())
#     else:
#         db._state_= {}

#     return db

# def funname(file):
#     """Return variable names from file names."""
#     if type(file) is str:
#         files = [file]
#     else:
#         files = file
#     bases = [os.path.basename(f) for f in files]
#     names = [os.path.splitext(b)[0] for b in bases]
#     if type(file) is str:
#         return names[0]
#     else:
#         return names

