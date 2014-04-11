import abc
import logging


output_registry = {}
LOG_LEVEL_NOISY = 15

class OutputMetaclass(abc.ABCMeta):
    def __init__(cls, name, b, d):
        abc.ABCMeta.__init__(cls, name, b, d)
        if name == "OutputBase":
            return
        if not name.endswith("Output"):
            raise ValueError("Output classes must be named [Name]Output")
        config_name = name[:-len("Output")].lower()

        output_registry[config_name] = cls
        if hasattr(cls, "_aliases"):
        	for alias in cls._aliases:
        		#Do not over-ride superclass aliases
        		if alias not in output_registry:
        			output_registry[alias] = cls

class OutputBase(object):
	__metaclass__ = OutputMetaclass

	def __init__(self):
		super(OutputBase,self).__init__()
		self._columns = []
		self.closed=False
		self.begun_sampling = False

	def log_debug(self, message, *args, **kwargs):
		logging.debug(message, *args, **kwargs)
	def log_noisy(self, message, *args, **kwargs):
		logging.log(LOG_LEVEL_NOISY, message, *args, **kwargs)
	def log_info(self, message, *args, **kwargs):
		logging.info(message, *args, **kwargs)
	def log_warning(self, message, *args, **kwargs):
		logging.warning(message, *args, **kwargs)
	def log_error(self, message, *args, **kwargs):
		logging.error(message, *args, **kwargs)
	def log_critical(self, message, *args, **kwargs):
		logging.critical(message, *args, **kwargs)

	@property
	def columns(self):
		return self._columns

	@columns.setter
	def columns(self, columns):
		self._columns = columns

	@columns.deleter
	def columns(self, columns):
		self._columns = []

	def add_column(self, name, dtype, comment=""):
		self.columns.append((name,dtype,comment))

	def del_column(self, column):
		if isinstance(column, int):
			self.columns.pop(column)
		elif isinstance(column, tuple):
			self.columns.remove(tuple)
		elif isinstance(column, basestring):
			self.columns.pop(self.column_index_for_name(column))
		else:
			raise TypeError("Unknown type of column to delete")

	def column_index_for_name(self, name):
		for i,(col_name,dtype,comment) in enumerate(self._columns):
			if name==col_name:
				return i

	@property
	def column_names(self):
		return [c[0] for c in self._columns]

	def parameters(self, params, extra=None):
		""" 
		Tell the outputter to save a vector of parameters.
		The number of parameters and their types
		must match what the columns are.
		"""
		#Check that the length is correct
		if self.closed:
			raise RuntimeError("Tried to write parameters to closed output")
		if extra:
			params = list(params[:])
			nstandard = len(params)
			for i in xrange(nstandard,len(self._columns)):
				name = self._columns[i][0]
				params.append(extra[name])
		if not len(params)==len(self._columns):
			raise ValueError("Sampler error - tried to save wrong number of parameters, or failed to set column names")
		#If this is our first sample then 
		if not self.begun_sampling:
			self._begun_sampling(params)
			self.begun_sampling=True
		self._write_parameters(params)

	def metadata(self, key, value, comment=""):
		"""
		Save an item of metadata, with an optional comment.
		For many some output types this will only work before
		sampling has begun.
		"""
		if self.closed:
			raise RuntimeError("Tried to write metadata info to closed output")
		self._write_metadata(key, value, comment)

	def final(self, key, value, comment=""):
		"""
		Save an item of final metadata (like a convergence test).
		This is designed to be run after sampling has finished;
		doing otherwise might cause problems.
		"""
		if self.closed:
			raise RuntimeError("Tried to write final info to closed output")
		self._write_final(key, value, comment)

	def close(self):
		self.closed=True
		self._close()

	#These are the methods that subclasses should
	#implement.  _begun_sampling and _close are optional.
	#The others are mandatory

	def _begun_sampling(self, params):
		pass

	def _close(self):
		pass

	@abc.abstractmethod
	def _write_parameters(self, params):
		pass

	@abc.abstractmethod
	def _write_metadata(self, key, value, comment):
		pass

	@abc.abstractmethod
	def _write_final(self, key, value, comment):
		pass

	@classmethod
	def from_options(cls, options):
		""" This method should create an output object from the section of ini file it is given"""
		raise NotImplemented("The format mode you tried to use is incomplete - sorry")

	@classmethod
	def load(cls, *args):
		""" This method should load back in data written using the format.
			It returns a triplet of (column_names, columns, metadata, final_metadata)
			with column_names a list of strings
			     columns a list of numpy arrays
			     metadata a dictionary
			     final_metadata a dictionary
		"""
		raise NotImplemented("The format mode you tried to use is incomplete - sorry")

