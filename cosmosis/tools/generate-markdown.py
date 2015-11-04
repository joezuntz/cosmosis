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

page_template = u"""
# {name} module

## {purpose}

**Name**: {name}

**File**: {filepath}

**Version**: {version}

**Author(s)**:

{author_list}

**URL**: {url}

**Cite**: 

{cite_list}

**Rules**: 

{rule_list}

**Assumptions**:

{assumption_list}

**Explanation**

{explanation}


##Parameters

These parameters can be set in the module's section in the ini parameter file.  
If no default is specified then the parameter is required.

Parameter | Description
------------|-------
{parameter_lines}

##Inputs

These parameters and data are inputs to the module, either supplied as parameters by the sampler or
computed by some previous module.  They are loaded from the data block.

Section | Parameter | Description
--------|-----------|-------
{input_lines}

##Outputs

These parameters and data are computed as outputs from the module

Section | Parameter | Description
--------|-----------|-------
{output_lines}


"""

#find the modules
outputs = collections.OrderedDict()
for dirpath, dirnames, filenames in os.walk('.'):
	if dirpath=='.': continue
	for filename in filenames:
		if filename=='module.yaml':
			category, module_name = os.path.split(dirpath)
			category = os.path.split(category)[1]
			if category=='.':continue
			sys.stderr.write("Found %s/%s\n"%(category,module_name))
			filepath = os.path.join(dirpath, filename)
			if category not in outputs:
				outputs[category] = collections.OrderedDict()
			info = ordered_load(open(filepath))
			name = info['name']
			version = info['version']
			info['full_name'] = '{}--{}'.format(name,version)
			info['page_name'] = '{}_{}'.format(name,version)
			if dirpath.startswith('./'): dirpath=dirpath[2:]
			info['dirname'] = dirpath
			info['explanation'] = info['explanation'].strip().strip('"').replace("\n","\n\n")
			outputs[category][info['full_name']] = info


f = open("wiki/default_modules.md","w")
for cat in outputs:
	f.write(u"## {}\n\n".format(cat))
	for mod in outputs[cat]:
		page_name = outputs[cat][mod]['page_name']
		purpose = outputs[cat][mod]['purpose']
		version = outputs[cat][mod]['version']
		name = outputs[cat][mod]['name']
		f.write(u"- [{} ({})](https://bitbucket.org/joezuntz/cosmosis/wiki/default_modules/{})  {}\n\n".format(name,version,page_name,purpose))
f.close()


def make_list(l):
	if l is None: return ""
	if isinstance(l, basestring):
		return u"- "+l
	else:
		return u"\n".join("- "+m for m in l)

def make_params(P):
	lines = []
	for section, params in P.items():
		for i,(name,desc) in enumerate(params.items()):
			sec = section if i==0 else ""
			lines.append(u"{}|{}|{}".format(sec,name,desc))
	return "\n".join(lines)

def make_page(info):
	parameter_lines = []
	for name,desc in info['params'].items():
		parameter_lines.append(u"{}|{}".format(name,desc))
	parameter_lines = '\n'.join(parameter_lines)
	input_lines = make_params(info['inputs'])
	output_lines = make_params(info['outputs'])
	markdown = page_template.format(
		author_list = make_list(info['attribution']),
		cite_list = make_list(info['cite']),
		rule_list = make_list(info['rules']),
		assumption_list = make_list(info['assumptions']),
		input_lines = input_lines,
		output_lines = output_lines,
		parameter_lines=parameter_lines,
		filepath=info['dirname'] + "/" + info['interface'],
		**info)
	open("wiki/default_modules/{}.md".format(info['page_name']),"w").write(markdown.encode("utf-8"))



for cat in outputs:
	for mod in outputs[cat]:
		try:
			make_page(outputs[cat][mod])
		except:
			print cat, mod
			raise