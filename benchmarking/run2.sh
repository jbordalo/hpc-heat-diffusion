#!/bin/bash

DELIMITER=";"

VERSION=$1
THREADS=$2
WPT=$3
RUNS=$4

OUTPUT=./results/$VERSION"_"$THREADS"_"$WPT.csv

rm $OUTPUT &>/dev/null

for (( i=0; i < $RUNS; i++ ))
do
	../$VERSION $THREADS $WPT > temp
	awk '/It took/{printf "%s", $(NF-1)}' temp >> $OUTPUT
	rm temp
	echo -n $DELIMITER >> $OUTPUT
done

sed -i '$ s/.$//' $OUTPUT
