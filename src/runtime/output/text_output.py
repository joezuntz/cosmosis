from output_base import OutputBase

class TextColumnOutput(OutputBase):
	def __init__(self, filename, delimiter='\t'):
		super(TextColumnOutput, self).__init__()
		self.delimiter=delimiter
		self._file = open(filename, 'w')
		self._metadata = {}

	def _close(self):
		self._file.close()

	def _begun_sampling(self, params):
		#write the name line
		name_line = '#'+self.delimiter.join(c[0] for c in self.columns) + '\n'
		self._file.write(name_line)
		#now write any metadata.
		#text mode does not support comments
		for (key,(value,comment)) in self._metadata.items():
			self._file.write('#{k}={v} #{c}\n'.format(k=key,v=value,c=comment))
		self._metadata={}

	def _write_metadata(self, key, value, comment=''):
		#We save the metadata until we get the first 
		#parameters since up till then the columns can
		#be changed
		#In the text mode we cannot write more metadata
		#after sampling has begun (because it goes at the top).
		#What should we do?
		self._metadata[key]= (value, comment)

	def _write_parameters(self, params):
		line = self.delimiter.join(str(x) for x in params) + '\n'
		self._file.write(line)

	def _write_final(self, key, value, comment=''):
		#I suppose we can put this at the end - why not?
		c=''
		if comment:
			c='  #'+comment
		self._file.write('#{k}={v} {c}\n'.format(k=key,v=value,c=c))

	@classmethod
	def from_ini(cls, ini):
		#look something up required parameters in the ini file.
		#how this looks will depend on the ini 
		filename = ini['filename']
		delimiter = ini.get('delimiter','\t')
		return cls(filename, delimiter=delimiter)

	@classmethod
	def load(cls, *args):
		filename = args[0]
		#Read the metadata
		for i,line in open(filename):
			line=line.strip()
			if line.startswith('#'):
				if i==0:
					column_names = line.lstrip('#').split()
				else:
					#parse form '#key=value #comment'
					pass


