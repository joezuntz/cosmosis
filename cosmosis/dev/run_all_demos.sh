#!/usr/bin/env bash

if [ -n "$1" ]
then
	start=$1
else
	start=1
fi

echo Starting from demo $start

# This is used in demo 10
export HALOFIT=halofit_takahashi

# Loop through all demos
for (( D=$start ; D<=24 ; D++ ))
do
	# There is no demo 23 yet
	if [[ $D == 23 ]]
	then
		continue
	fi

	# Check main run
	cosmosis demos/demo$D.ini
	if [ $? -ne 0 ]
		then
		echo "Error running Demo $D"
		exit 1
	fi

	# Also check we can postprocess okay
	postprocess demos/demo$D.ini -o output/plots -p demo$D
	if [ $? -ne 0 ]
		then
		echo "Error postprocessing Demo $D"
		exit 1
	fi
done

# Now do some MPI test runs

# Test one python MPI sampler and one fortran
mpirun -n 3 cosmosis --mpi demos/demo5.ini
if [ $? -ne 0 ]
	then
	echo "Error running Demo 3 in MPI mode"
	exit 1
fi

mpirun -n 3 cosmosis --mpi demos/demo9.ini
if [ $? -ne 0 ]
	then
	echo "Error running Demo 3 in MPI mode"
	exit 1
fi


exit 0


