import section_names
import string
import hashlib

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

TRUE_STRINGS =  ["T","t","True","TRUE","true","y","Y","yes","Yes","YES","1"]
FALSE_STRINGS = ["F","f","False","FALSE","false","n","N","no","No","NO","0"]

def boolean_string(s):
	if s is True or s is False:
		return s
	if not isinstance(s,basestring):
		raise ValueError("Non-string passed to boolean_string for conversion to bool")
	if s in TRUE_STRINGS:
		return True
	elif s in FALSE_STRINGS:
		return False
	raise ValueError("Could not convert string '%s' to True/False value"%s)


def try_numeric(x):
	if isinstance(x, int) or isinstance(x, float):
		return x
	try:
		return int(x)
	except:
		pass
	try:
		return float(x)
	except:
		pass
	return x

def letter_combinations():
	""" Yield an iterator that produces strings A,B,C,D,...,X,Y,Z,AA,AB,AC,...AY,AZ,BA,...ZY,ZZ,AAA,..."""
	for n in xrange(1,26+1):
		for letters in itertools.combinations_with_replacement(string.uppercase, n):
			yield ''.join(letters)


def blind_parameter_offset_from_name(section, name):
	#normalize the name
	section = getattr(section_names.section_names, section, section)
	phrase = (section+name).upper()
	#hex number derived from code phrase
	m = hashlib.md5(phrase).hexdigest()
	#convert to decimal
	s = int(m, 16)
	# last 8 digits
	f = s%100000000
	# turn 8 digit number into value between 0 and 1 and return
	g = f*1e-8
	return g



class EverythingIsNan(object):
	def __getitem__(self, param):
		return np.nan
everythingIsNan = EverythingIsNan()
