import section_names

class FriendlyNameLookup(object):
	def __init__(self):
		self.codes={}
		for name in dir(section_names.section_names):
			if name.startswith("__"): continue
			value = getattr(section_names.section_names, name)
			self.codes[value] = name
	def __getitem__(self, code):
		return self(code)
	def __call__(self, code):
		try:
			return self.codes[code]
		except KeyError:
			return code
section_friendly_names = FriendlyNameLookup()

class EverythingIsNan(object):
	def __getitem__(self, param):
		return np.nan
everythingIsNan = EverythingIsNan()
