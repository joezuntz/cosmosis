from output_base import OutputBase
import utils
import numpy as np
import os

CHAIN_NAME = '_%d.txt'
PARAM_NAME = '.paramnames'


class CosmoMCOutput(OutputBase):
	def __init__(self, dirname, filename, rank=0, nchain=1, delimiter='\t'):
		super(CosmoMCOutput, self).__init__()
		self.delimiter=delimiter
		self._dirname = dirname
		self._filename = filename
		self.rank = rank
        if nchain > 1:
			self._file = open(os.path.join(self._dirname,''.join([self._filename, str(CHAIN_NAME % (self.rank+1))]) ), 'w')
		else:
			self._file = open(os.path.join(self._dirname,''.join([self._filename, '.txt']) ), 'w')
		if self.rank == 0: 
			self._paramfile = open(os.path.join(self._dirname,''.join([self._filename, str(PARAM_NAME)]) ), 'w')
		self._metadata = {}

	def _close(self):
		self._file.close()

	def _begun_sampling(self, params):
		#write the name line
		if self.rank == 0:
			for c in self.columns[2:]:
				self._paramfile.write('\t '.join([c[0],'\n']) )
			self._paramfile.close()
		self._metadata={}

	def _write_metadata(self, key, value, comment=''):
		#We save the metadata until we get the first 
		#parameters since up till then the columns can
		#be changed
		#In the text mode we cannot write more metadata
		#after sampling has begun (because it goes at the top).
		#What should we do?
		self._metadata[key] = (value, comment)

	def _write_comment(self, comment):
		#Do not think cosmomc can handle comments
		pass

	def _write_parameters(self, params):
		line = self.delimiter.join(('%.7E'%x) for x in params) + '\n'
		multiplicity = 1.0
		line = ' %f   %s' % (multiplicity,line)
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
		dirname = options.get('dirname', '.')
		filename = options['filename']
		delimiter = options.get('delimiter', '\t')
		rank = options.get('rank', 1)
		nchain = options.get('parallel', 1)
		return cls(dirname, filename, rank, nchain, delimiter=delimiter)

	@staticmethod
	def parse_value(x):
		x = utils.try_numeric(x)
		if x=='True':
			x=True
		if x=='False':
			x=False
		return x


	@classmethod
	def load(cls, *args):
		filename = args[0]
		#Read the metadata
		started_data = False
		metadata = {}
		final_metadata = {}
		data = []
		for i,line in enumerate(open(filename)):
			line=line.strip()
			if not line: continue
			if line.startswith('#'):
				line=line.lstrip('#')
				if i==0:
					column_names = line.split()
				else:
					#parse form '#key=value #comment'
					if line.count('#')==0:
						key_val = line.strip()
						comment = ''
					else:
						key_val, comment = line.split('#', 1)
					key,val = key_val.split('=',1)
					val = cls.parse_value(val)
					if started_data:
						final_metadata[key] = val
					else:
						metadata[key] = val
			else:
				started_data = True
				words = line.split()
				vals = [float(word) for word in words]
				data.append(vals)
		data = np.array(data).T
		cols = [col for col in data]
		for i in xrange(len(cols)):
			if (cols[i]==cols[i].astype(int)).all():
				cols[i] = cols[i].astype(int)
		return column_names, cols, metadata, final_metadata

	@staticmethod
	def load_txt_tables(*args):
		filename = args[0]	
		t = np.loadtxt(filename,dtype=int).T
		return t 

	@staticmethod
	def get_chains(*args):
		dir_name = args[0]	
		chains = []
		try:
			content = os.listdir(dir_name)
			for c in content:
				if c.startswith(CHAIN_NAME[:-7]):
					chains.append(''.join([dir_name,'/',c]))
		except:
			pass
		chains.sort()
		return chains


