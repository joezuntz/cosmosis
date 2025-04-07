from .elements import PostProcessorElement
from cosmosis.runtime.config import Inifile
import os
import tempfile
import sys
from collections import defaultdict

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
	prior_lines = []
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
		elif line=='START_OF_PRIORS_INI':
			state='priors'
			continue
		elif line=='END_OF_PRIORS_INI':
			state=None
			continue
		if state=='params':
			param_lines.append(line+"\n")
		elif state=='values':
			value_lines.append(line+"\n")
		elif state=='priors':
			prior_lines.append(line+"\n")

	f = Inifile(None)
	f.read_string(str("\n".join(param_lines)))
	g = Inifile(None)
	g.read_string(str("\n".join(value_lines)))
	h = Inifile(None)
	h.read_string(str("\n".join(prior_lines)))

	return f, g, h




#


class Rerunner(PostProcessorElement):
	def __init__(self, dirname, *args, **kwargs):
		self.rerun_dirname=dirname
		super(Rerunner, self).__init__(*args, **kwargs)

	def test_run_sample(self, sample):

		#Turn the output header into an ini file.
		#Definitely better to do it here as any
		#envar substitution is already done
		params_ini, values_ini, priors_ini = ini_from_header(self.source.comments[0])
		params_file = tempfile.NamedTemporaryFile(suffix='.ini', mode='w+')
		values_file = tempfile.NamedTemporaryFile(suffix='.ini', mode='w+')
		priors_file = tempfile.NamedTemporaryFile(suffix='.ini', mode='w+')

		#Now override the old parameters so that we:
		# - run the test sampler not the old sampelr
		# - save to our temp dir
		# - read from our temp files

		try:
			params_ini.add_section('test')
		except:
			pass

		likes = params_ini.get('pipeline', 'likelihoods', fallback='')
		params_ini.set('pipeline', 'values', values_file.name)
		params_ini.set('pipeline', 'priors', priors_file.name)
		params_ini.set('pipeline', 'timing', 'T')
		params_ini.set('pipeline', 'debug', 'T')
		params_ini.set('runtime', 'sampler', 'test')
		params_ini.set('test', 'save_dir', self.rerun_dirname)


		params_ini.write(params_file)
		params_file.flush()

		nvaried = self.source.metadata[0]['n_varied']
		varied_params = self.source.colnames[:nvaried]
		values_update = []
		i=0
		print("Updating these parameters:")
		for (sec,key), val in values_ini:
			if '{}--{}'.format(sec,key) in varied_params:
				l, _, u = val.split()
				print("    {}--{} = {}".format(sec,key,sample[i]))
				value_string = "{}  {}  {}".format(l, sample[i], u)
				values_update.append((sec, key, value_string))
				i+=1
			else:
				values_update.append((sec, key, val))
		print("")
		for sec, key, val in values_update:
			values_ini.set(sec, key, val)

		values_ini.write(values_file)
		values_file.flush()


		# No changes needed to prior
		priors_ini.write(priors_file)
		priors_file.flush()

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
		cmd = "{0} cosmosis {1}".format(new_ld, params_file.name)
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
