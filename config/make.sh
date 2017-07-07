#!/bin/bash
# This trivial script is just so that the "shifter"
# make command has a single executable to run.
# You can ignore it if that means nothing to you.
cd $COSMOSIS_SRC_DIR

echo "Starting make at" `date`
make clean
make
echo "Ending make at" `date`
