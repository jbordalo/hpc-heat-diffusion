#!/bin/bash

DELIMITER=";"

VERSION=$1
RUNS=$2

OUTPUT=./results/$VERSION.csv

rm $OUTPUT &>/dev/null

for (( i=0; i < $RUNS; i++ ))
do
	../$VERSION > temp
	awk '/It took/{printf "%s", $(NF-1)}' temp >> $OUTPUT
	rm temp
	echo -n $DELIMITER >> $OUTPUT
done

sed -i '$ s/.$//' $OUTPUT
