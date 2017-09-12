import os
import yaml
import glob
from collections import OrderedDict
import tabulate

#Generate the summary page of all the samplers
#Generate the specific page for each sampler.

def name_for_sampler_page(name):
	return name

markdown_template = u"""
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

rst_template = u"""The {name} sampler
------------------

{purpose}

{header_table}

{explanation}

Installation
============

{installation}


Parameters
============

These parameters can be set in the sampler's section in the ini parameter file.  
If no default is specified then the parameter is required. A listing of "(empty)" means a blank string is the default.

.. list-table::
    :widths: auto
    :header-rows: 1
{parameter_table}
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


def header_table(info):
	table = [
		["Name", info["name"]],
		["Version", info["version"]],
		["Author(s)", info["authors"]],
		["URL", info["url"]],
		["Citation(s)", info["citations"]],
		["Parallelism", info["parallel"]],
	]
	return tabulate.tabulate(table, tablefmt='rst')



def make_parameter_table(params):
	headers = ['Parameter', 'Type', 'Meaning', 'Default']
	text = """
    * - Parameter
      - Type
      - Meaning
      - Default
"""
	table = []
	for pname, description in list(params.items()):
		try:
			dtype, default, rest = parse_parameter_description(description)
			table.append([pname,dtype,rest,default])
			text += """    * - {}
      - {}
      - {}
      - {}
""".format(pname,dtype,default,rest)
		except (IndexError, ValueError):
			print "ERROR: Could not parse in {0}".format(name)
			print description
	return text

#	return tabulate.tabulate(table, headers=headers, tablefmt='rst')


def generate_sampler_wiki(info):
	"Generate wiki markdown for a single sampler"
	info = info.copy()
	name = info['name']
	info['explanation'] = info['explanation'].replace("\n",'\n\n').strip('"')
	info['installation'] = info['installation'].replace("\n",'\n\n').strip('"')
	info['authors'] = ','.join(info['attribution'])
	info['citations']=', '.join(info['cite'])
	info['header_table'] = header_table(info)
	page_name = name_for_sampler_page(name)
	page = open('doc/reference/samplers/{}.rst'.format(page_name), 'w')
	info['parameter_table'] = make_parameter_table(info['params'])
	info['name'] = name.capitalize()
	markdown = rst_template.format(**info)
	page.write(markdown)
	page.close()



def generate_overview(infos):
	pass


def generate_links(infos):
	f = open("doc/reference/samplers/samplers.rst",'w')
	f.write("""
Samplers
--------

Samplers are the different methods that CosmoSIS uses to choose points in parameter spaces to evaluate.

Some are designed to actually explore likelihood spaces; others are useful for testing and understanding likelihoods.

.. toctree::
    :maxdepth: 1

""")

	for info in infos:
		name = info['name']
		page = name_for_sampler_page(name)
		slug = info['purpose']
		f.write("     {}: {} <{}>\n".format(name, slug, page))


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