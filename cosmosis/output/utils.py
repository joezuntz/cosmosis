import os

TRUE_STRINGS =  ["T","t","True","TRUE","true","y","Y","yes","Yes","YES","1"]
FALSE_STRINGS = ["F","f","False","FALSE","false","n","N","no","No","NO","0"]

def boolean_string(s):
	if s is True or s is False:
		return s
	if not isinstance(s,str):
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

def parse_value(x):
    x = try_numeric(x)
    if x=='True':
        x=True
    if x=='False':
        x=False
    return x

def file_exists_and_is_empty(fpath):
    """
    
    from https://stackoverflow.com/questions/2507808/how-to-check-whether-a-file-is-empty-or-not
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) == 0

def file_exists_and_is_not_empty(fpath):
    """
    
    from https://stackoverflow.com/questions/2507808/how-to-check-whether-a-file-is-empty-or-not
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
