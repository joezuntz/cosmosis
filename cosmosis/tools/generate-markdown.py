"""
This script, when run from the cosmosis-standard-library directory, 
scans for module.yaml files and uses them to generate the markdown 
for the wiki page documenting the standard modules.

It is not intended for users.

"""
import yaml
import os
import collections
import sys
import codecs
UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

#From stack overflow!
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=collections.OrderedDict):
	class OrderedLoader(Loader):
		pass
	OrderedLoader.add_constructor(
		yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
		lambda loader, node: object_pairs_hook(loader.construct_pairs(node)))
	return yaml.load(stream, OrderedLoader)


outputs = collections.OrderedDict()
for dirpath, dirnames, filenames in os.walk('.'):
	if dirpath=='.': continue
	for filename in filenames:
		if filename=='module.yaml':
			category, module_name = os.path.split(dirpath)
			category = os.path.split(category)[1]
			sys.stderr.write("Found %s/%s\n"%(category,module_name))
			filepath = os.path.join(dirpath, filename)
			if category not in outputs:
				outputs[category] = collections.OrderedDict()
			outputs[category][module_name] = ordered_load(open(filepath))


def print_part(name, info):
	print u'_%s_' % name.capitalize()
	print
	section = info[name]
	for (section_name, params) in section.items():
		print u'- %s' % section_name
		print
		for (param, meaning) in params.items():
			print u'    * %s: %s' % (param, meaning)
			print
	print

def print_params(info):
	print u'_Parameters_:'
	print
	for (param, meaning) in info['params'].items():
		print u'- %s: %s' % (param, meaning)
		print
	print

for cat in outputs:
	print u'# [%s](https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules#%s)'%(cat,cat)
	print
	for mod in outputs[cat]:
		print u'- [%s](https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules#markdown-header-%s)'%(mod,mod)
		print
	print


for cat in outputs:
	print u'# %s' % (cat, )
	print
	for mod in outputs[cat]:
		print u'## %s'%(mod,)
		print
		info = outputs[cat][mod]
		print info['explanation'].strip('"')
		print_params(info)
		print_part('inputs', info)
		print_part('outputs', info)

