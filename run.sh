#!/bin/bash

bs=${1:-8}
n=${2:-8}

for tn in 1 4 7 14 28 32 48 52 54 56; do
    ./bin/cpuft -n $n -bs $bs -t $tn -numa
done

