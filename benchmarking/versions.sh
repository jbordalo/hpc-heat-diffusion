#!/bin/bash

for ver in "cuda" "cuda-shared"
do
	for thread in 4 8 16 32
	do
	  echo "Doing version $ver_$thread"
	  ./run.sh $ver $thread 10;
  done
done


for ver in "cuda-1D" "cuda-1D-shared"
do
	for thread in 16 32 64 128 256 512 1024
	do
	  echo "Doing version $ver_$thread"
	  ./run.sh $ver $thread 10;
  done
done

for ver in "cuda-wpt"
do
	for work in 2 3
	do
	  echo "Doing version $work"
	  ./run2.sh $ver 8 $work 10;
  done
done

for ver in "cuda-1D-wpt"
do
	for work in 2 4 8
	do
	  echo "Doing version $work"
	  ./run2.sh $ver 32 $work 10;
	  ./run2.sh $ver 64 $work 10;
  done
done

for stream in 2500 5000 10000 20000
do
  echo "Doing version $stream"
  ./run2.sh cuda-1D-shared-stream 32 $stream 10;
done
