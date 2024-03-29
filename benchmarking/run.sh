#!/bin/bash

DELIMITER=";"

VERSION=$1
THREADS=$2
RUNS=$3

OUTPUT=./results/$VERSION"_"$THREADS.csv

rm $OUTPUT &>/dev/null

for (( i=0; i < $RUNS; i++ ))
do
	../$VERSION $THREADS > temp
	awk '/It took/{printf "%s", $(NF-1)}' temp >> $OUTPUT
	rm temp
	echo -n $DELIMITER >> $OUTPUT
done

sed -i '$ s/.$//' $OUTPUT
