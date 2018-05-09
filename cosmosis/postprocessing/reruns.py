from __future__ import print_function
from builtins import zip
from .elements import PostProcessorElement
from cosmosis.runtime.config import Inifile
import os
import tempfile


def ini_from_header(header_text):
	"""
	Parse a cosmosis-format set of header comments that encode the
	parameters and values used in a run.

	These lines use the sentinels:
	START_OF_PARAMS_INI
	END_OF_PARAMS_INI
	START_OF_VALUES_INI
	END_OF_VALUES_INI
	to mark sections.

	"""
	state=None
	param_lines = []
	value_lines = []
	for line in header_text:
		line=line.strip()
		if line=='START_OF_PARAMS_INI':
			state='params'
			continue
		elif line=='END_OF_PARAMS_INI':
			state=None
			continue
		elif line=='START_OF_VALUES_INI':
			state='values'
			continue
		elif line=='END_OF_VALUES_INI':
			state=None
			continue
		if state=='params':
			param_lines.append(line+"\n")
		elif state=='values':
			value_lines.append(line+"\n")

	f = tempfile.NamedTemporaryFile(suffix='.ini')
	f.writelines(param_lines)
	f.flush()

	g = tempfile.NamedTemporaryFile(suffix='.ini')
	g.writelines(value_lines)
	g.flush()

	return f, g




#


class Rerunner(PostProcessorElement):
	def __init__(self, dirname, *args, **kwargs):
		self.rerun_dirname=dirname
		super(Rerunner, self).__init__(*args, **kwargs)

	def test_run_sample(self, sample):

		#Turn the output header into an ini file.
		#Definitely better to do it here as any
		#envar substitution is already done
		param_ini, value_ini = ini_from_header(self.source.comments[0])


		#Now override the old parameters so that we:
		# - run the test sampler not the old sampelr
		# - save to our temp dir
		# - read from our temp files
		param_ini.write("[pipeline]\nvalues={}\n".format(value_ini.name))
		param_ini.write("[runtime]\nsampler=test\n[test]\nsave_dir={}\nquiet=F\ndebug=T\n".format(self.rerun_dirname))
		param_ini.flush()

		# and the values so that we use the desired parameters
		nvaried = self.source.metadata[0]['n_varied']
		for i,(name, value) in enumerate(zip(self.source.colnames, sample)):
			if i>=nvaried:
				break
			if '--' in name:
				section, key = name.split('--', 1)
				value_ini.write("[{}]\n{}={}\n".format(section, key, value))
		value_ini.flush()

		#A bit of painful fiddling to work around
		#link issues on a mac.  To run the postprocess
		#command we had to drop a problematic entry
		#in DYLD_LIBRARY_PATH.  But to run cosmosis
		#we need it back!
		if "UPS_DIR" in os.environ and os.uname()[0]=='Darwin':
			extra_ld = '{0}/lib'.format(os.environ['GCC_FQ_DIR'])
			old_ld = os.environ['DYLD_LIBRARY_PATH']
			new_ld = 'DYLD_LIBRARY_PATH="{0}:{1}"'.format(old_ld, extra_ld)
		else:
			new_ld = ''

		#Run the actual command
		cmd = "{0} cosmosis {1}".format(new_ld, param_ini.name)
		status = os.system(cmd)

		#Now we have the cosmology data!
		return status




class BestFitRerunner(Rerunner):
	"Re-run sample(s) from an existing chain under the test sampler"

	def run(self):
		best_fit_index = self.source.get_col("post").argmax()
		sample = self.source.get_row(best_fit_index)
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		print("Re-running maximum-posterior sample and saving results to {}".format(self.rerun_dirname))
		print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
		self.test_run_sample(sample)


		return []
