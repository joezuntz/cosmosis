# This file must be sourced, so that it can establish the appropriate
# environment for a development session.
# Note: this currently only supports bash

# detect COSMOSIS_SRC_DIR
cosmosis_dir="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"
cosmosis_dir=${cosmosis_dir%/config}

# allow user to override UPS directory
if [ "$#" != "0" ]; then
  product_db=$1
  if [ !  -d "$product_db" ]
  then
    echo "The directory $product_db does not exist."
    return 1
  fi
else
  product_db=`cat $cosmosis_dir/config/ups`  
fi

if [ ! -f "$product_db/setups" ]
then
    echo "The directory $product_db does not appear to contain the UPS products."
    return 1
fi

export COSMOSIS_SRC_DIR="$cosmosis_dir"

# initialize UPS
source $product_db/setups
if [ -z "$PRODUCTS" ]
then
    echo "The setup of the UPS system has failed; please ask a local expert for assistance."
    return 1
fi

# Set the library path appropriate for our flavor.
libdir=${COSMOSIS_SRC_DIR}/cosmosis/datablock
flavor=$(ups flavor -1)
if [ "$flavor" == "Darwin64bit" ]
then
    export DYLD_LIBRARY_PATH=${libdir}:$DYLD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=${libdir}:$LD_LIBRARY_PATH
fi

export PATH=${COSMOSIS_SRC_DIR}/bin:$PATH
export PYTHONUSERHOME=${COSMOSIS_SRC_DIR}
export PYTHONPATH=${COSMOSIS_SRC_DIR}:$PYTHONPATH

# setup UPS packages
setup -B scipy v0_13_0b -q +e5:+prof
setup -B gsl v1_16 -q +prof
setup -B cfitsio v3_35_0 -q +prof
setup -B pyfits v3_2a -q +e5:+prof
setup -B pyyaml v3.11 -q +e5:+prof
setup -B wmapdata v5_00

export PS1="(cosmosis) $PS1"
