class PostProcessorElement(object):
	def __init__(self, data_source, **options):
		self.source = data_source
		self.options = {}
		self.options.update(options)

	def run(self):
		print "I do not know how to produce some results for this kind of data"
		return []

	def finalize(self):
		pass
