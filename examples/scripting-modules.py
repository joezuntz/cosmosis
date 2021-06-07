import numpy as np
import os

# Sorry for the awkward imports - this is improved in the next
# version.
from cosmosis.runtime.module import Module
from cosmosis.runtime.config import Inifile
from cosmosis.runtime.pipeline import config_to_block
from cosmosis.datablock import DataBlock

# root directory
csd = os.environ['COSMOSIS_SRC_DIR'] + '/'

# Make an inifile object.
# We could also just load this from an ini file by replacing
# None with the file name.
ini = Inifile(None)
ini.add_section('camb')

# options for camb, copied from demo 1, except I had to manually put
# in the variable csd for $COSMOSIS_SRC_DIR
options = dict(
    file = "cosmosis-standard-library/boltzmann/camb/camb.so",
    mode = "all",
    lmax = 2600,          #max ell to use for cmb calculation
    feedback = 2,         #amount of output to print
    accuracy_boost = 1.1, #CAMB accuracy boost parameter
    high_acc_default = True, #high accuracy is required w/ Planck data
    kmax = 100.0,       #max k - matter power spectrum
    zmin = 0.0,         #min value to save P(k,z)
    zmax = 1.0,         #max value to save P(k,z) 
    nz = 20,            #number of z values to save P(k,z) 
    do_tensors = False,   #include tensor modes
    do_lensing = True,    #lensing is required w/ Planck data
    high_ell_template = csd + 'cosmosis-standard-library/boltzmann/camb/camb_Jan15/HighLExtrapTemplate_lenspotentialCls.dat',
    matter_power_lin_version = 3, # Sum of required P(k,z) outputs:
)

# Convert our dictionary to an inifile object
for key, value in options.items():
    ini.set('camb', key, str(value))

# Load the module.  This does not configure it yet
camb = Module.from_options('camb', ini, root_directory=csd)

# One way you can make a block - from an ini file.
config_block = config_to_block(['camb'], ini)

# Now configure the module by running the setup function we found
camb.setup(config_block, quiet=True)

# Another way to make a block - from yaml string
values_block = DataBlock.from_string("""
cosmological_parameters:
    Omega_b: 0.04
    Omega_c: 0.26
    Omega_lambda: 0.7
    Omega_nu: 0.0
    Omega_k: 0.0
    hubble: 68.0
    n_s: 0.96
    A_s: 2.1e-9
    tau: 0.08
""")

# Run the module on the params above
status = camb.execute(values_block)

if status != 0:
    raise RuntimeError("Failed to run CAMB module.")

# Now some examples of getting results out:

# Pull out a scalar value
age = values_block['distances','age']

# You can also pull grids out using this method, which makes sure
# the ordering of the 2D grid is what you ask for
z, k, P = values_block.get_grid('matter_power_lin', 'z' , 'k_h', 'P_k')
P = P[0]  # get z=0 part
# Do a log interp into P(k) at k = 0.1 Mpc/h
P_01 = np.exp(np.interp(np.log(0.1), np.log(k), np.log(P) ))

# Some results
print("\n\n")
print("Age = {} Gyr".format(age))
print("P(k=0.1, z=0) = {:.1f} (Mpc/h)^3".format(P_01))
print("\n")