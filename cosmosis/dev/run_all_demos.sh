#!/usr/bin/env bash
for (( D=1 ; D<=19 ; D++ ))
do
	cosmosis demos/demo$D.ini
	if [ $? -ne 0 ]
		then
		echo "Error running Demo $D"
		exit 1
	fi
	postprocess demos/demo$D.ini -o plots -p demo$D
	if [ $? -ne 0 ]
		then
		echo "Error postprocessing Demo $D"
		exit 1
	fi


done

