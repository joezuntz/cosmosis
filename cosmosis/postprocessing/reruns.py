from .elements import PostProcessorElement
from cosmosis.runtime.config import Inifile
import os
import tempfile

class FakeFile(object):
	def __init__(self, iterable):
		self.it=iterable
		self.i=0
	def read(self):
		pass
	def open(self, *args, **kwargs):
		return self
	def close(self, *args, **kwargs):
		pass
	def __iter__(self):
		return iter(self.it)
	def readline(self):
		try:
			x = self.it[self.i]
		except IndexError:
			x = ''
		self.i += 1
		return x

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
	save_dir = "rerun_output_data"
	def test_run_sample(self, sample, temp_params, temp_values, temp_dir):

		#Turn the output header into an ini file.
		#Definitely better to do it here as any
		#envar substitution is already done
		param_ini, value_ini = ini_from_header(self.source.comments[0])


		#Now override the old parameters so that we:
		# - run the test sampler not the old sampelr
		# - save to our temp dir
		# - read from our temp files
		param_ini.write("[pipeline]\nvalues={}\n".format(temp_values))
		param_ini.write("[runtime]\nsampler=test\n[test]\nsave_dir={}\n".format(self.save_dir))
		param_ini.flush()

		# and the values so that we use the desired parameters
		for (name, value) in zip(self.source.colnames, sample):
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
		#Save to these temporaries
		temp_params = "temp.ini"
		temp_values = "temp_values.ini"
		temp_dir = "temp_save"

		#
		self.test_run_sample(sample, temp_params, temp_values, temp_dir)

		#The directory temp_dir now contains
		#all the info we need

		return []
