# This file must be sourced, so that it can establish the appropriate
# environment for a development session. Source it in a shell in which
# your current working directory is the top-level directory for your
# build.

USAGE="Usage: source setup_for_dev.sh path-to-cosmosis-source"

if [ "$#" == "0" ]; then
    echo $USAGE
    return 1
fi

cosmosis_dir=$1

if [ !  -d "$cosmosis_dir" ]
then
    echo "The directory $cosmosis_dir does not exist"
    return 1
fi

if [ ! -f "$cosmosis_dir/config/rules.mk" ]
then
    echo "The directory $cosmosis_dir does not appear to contain the CosmoSIS source code"
    return 1
fi

export COSMOSIS_DIR="$cosmosis_dir"
export SOURCE_DIR=$COSMOSIS_DIR/src
export BUILD_TOP=$(pwd -P)

setup -B scipy v0_13_0b -q +e5:+prof

echo "You are ready to build."
echo "Use \"make -f \${COSMOSIS_DIR}/src/Makefile build\" and then \"./build\""
echo "Use \"./build test\" to execute all the tests"
echo "Any flags you supply to \"build\" are passed to \"make\""



