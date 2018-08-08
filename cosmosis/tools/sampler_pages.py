import os
import yaml
import glob
from collections import OrderedDict

#Generate the summary page of all the samplers
#Generate the specific page for each sampler.

def name_for_sampler_page(name):
	return name

page_template = u"""
# The {name} sampler

## {purpose}

Name: **{name}**

Version: **{version}**

Author(s): **{authors}**

URL: **{url}**

Cite: ** {citations} **

Parallel: **{parallel}**


{explanation}

##Installation

{installation}


##Parameters

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

Parameter | Type | Meaning | Default
------------|------|---------|--------
{parameter_lines}

"""

def parse_parameter_description(desc):
	paren, rest = desc.split(')', 1)
	paren = paren.strip()
	rest = rest.strip().strip('"')
	paren=paren.lstrip('(')
	if ';' in paren:
		dtype, default=paren.split(';')
		default=default.split('=')[1]
	else:
		dtype=paren
		default=""
	if default.strip()=="''":
		default = '(empty)'
	return dtype, default, rest



def generate_sampler_wiki(info):
	"Generate wiki markdown for a single sampler"
	info = info.copy()
	name = info['name']
	info['explanation'] = info['explanation'].replace("\n",'\n\n').strip('"')
	info['installation'] = info['installation'].replace("\n",'\n\n').strip('"')
	page_name = name_for_sampler_page(name)
	page = open('wiki/{}.md'.format(page_name), 'w')
	parameter_lines = []
	for pname,description in list(info['params'].items()):
		try:
			dtype, default, rest = parse_parameter_description(description)
			parameter_lines.append("{}|{}|{}|{}".format(pname, dtype, rest, default))
		except (IndexError, ValueError):
			print "ERROR: Could not parse in {0}".format(name)
			print description
			continue
		
	parameter_lines = '\n'.join(parameter_lines)
	info['name'] = name.capitalize()
	markdown = page_template.format( 
		citations=', '.join(info['cite']),
		authors = ','.join(info['attribution']),
		parameter_lines=parameter_lines,
		**info
		)
	page.write(markdown)
	page.close()



def generate_overview(infos):
	pass


def generate_links(infos):
	for info in infos:
		name = info['name']
		page = name_for_sampler_page(name)
		slug = info['purpose']
		print(" - [{0} sampler](samplers/{1}) {2}".format(page, name, slug))


def main():
	#get the base dir to work from
	src=os.environ['COSMOSIS_SRC_DIR']
	sampler_dir=os.path.join(src, "cosmosis", "samplers")
	#Find and parse all the files
	search_path = "{}/*/sampler.yaml".format(sampler_dir)
	yaml_files = glob.glob(search_path)
	infos = [yaml.load(open(f)) for f in yaml_files]
	#Make the ordering the same every time
	try:
		os.mkdir('wiki')
	except OSError:
		pass
	generate_links(infos)
	for info in infos:
		generate_sampler_wiki(info)


if __name__ == '__main__':
	main()