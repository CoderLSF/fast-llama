#!/bin/bash

workdir=$(dirname $(readlink -f $0))
cd $workdir
echo workdir:$workdir

CC=g++

build_flags=${1:-"-O3"}

CFLAGS="-mavx512f -mavx512bw -mavx512vl -mavx512dq -D_GNU_SOURCE -Wall -I ${workdir}/src"
CXXFLAGS="-std=c++20"
LIBS="-pthread -lm -lnuma"

TARGET=$workdir/main

srcs=$(find $workdir/src | egrep '\.cpp$')

includes=$(find $workdir/ | egrep '\.(h|hpp)' | xargs dirname | sort | uniq | awk '{printf("-I%s ", $0);}')


rm -f $TARGET
cmd="$CC -o $TARGET $srcs $CXXFLAGS $CFLAGS $LIBS $includes $build_flags "
echo $cmd
$cmd

if [ -e $TARGET ]; then
    echo -e $'\x1b[32mDone\x1b[0m'
else
    echo -e $'\x1b[31mFailed\x1b[0m'
fi



