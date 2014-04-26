# This file must be sourced, so that it can establish the appropriate
# environment for a development session. Source it in a shell in which
# your current working directory is the top-level directory for your
# build.

USAGE="Usage: source setup_for_dev.sh path-to-UPS-product-directory"

if [ "$#" == "0" ]; then
    echo $USAGE
    return 1
fi

product_db=$1

if [ !  -d "$product_db" ]
then
    echo "The directory $product_db does not exist."
    return 1
fi

if [ ! -f "$product_db/setups" ]
then
    echo "The directory $product_db does not appear to contain the UPS products."
    return 1
fi

export COSMOSIS_SRC_DIR="$PWD"

libdir=${COSMOSIS_SRC_DIR}/cosmosis/datablock
#if [ ! -d ${libdir} ]
#then
#    mkdir -p ${libdir}
#fi
#
#if [ ! -d ${libdir} ]
#then
#    echo "Failed to create the 'lib' directory under $PWD"
#    echo "Perhaps the directory is not writable."
#    return 1
#fi

source $product_db/setups
if [ -z "$PRODUCTS" ]
then
    echo "The setup of the UPS system has failed; please ask a local expert for assistance."
    return 1
fi

# Set the library path appropriate for our flavor.

flavor=$(ups flavor -1)
if [ "$flavor" == "Darwin64bit" ]
then
    export DYLD_LIBRARY_PATH=${libdir}:$DYLD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=${libdir}:$LD_LIBRARY_PATH
fi

export PATH=${COSMOSIS_SRC_DIR}/bin:$PATH
export PYTHONPATH=${COSMOSIS_SRC_DIR}:$PYTHONPATH

setup -B scipy v0_13_0b -q +e5:+prof
setup -B gsl v1_16 -q +prof
setup -B cfitsio v3_35_0 -q +prof
setup -B pyfits v3_2a -q +e5:+prof
setup -B wmapdata v5_00
