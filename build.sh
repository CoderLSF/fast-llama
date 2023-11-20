#!/bin/bash

workdir=$(dirname "$0")
if ! cd $workdir; then
    echo "Failed to enter program directory" 1>& 2
    exit 1
fi
CC=g++

build_flags=${1:-"-O3"}

DEFINES=""
if (uname -a | grep -q arm64); then
    ARCH_FLAGS='-march=armv8-a+simd -mcpu=native -mfpu=neon'
    ARCH_LIBS="-lpthread -lm"
    DEFINES="$DEFINES -DDISABLE_NUMA"
elif (uname -a | grep -q x86_64); then
    DEFINES="$DEFINES -D_GNU_SOURCE"
    if egrep -qi '^flags.*\<avx512f\>' /proc/cpuinfo; then
        ARCH_FLAGS="-march=native -mavx512f -mavx512bw -mavx512vl -mavx512dq"
    elif egrep -qi '^flags.*\<avx2\>' /proc/cpuinfo; then
        ARCH_FLAGS="-mavx2"
    elif egrep -qi '^flags.*\<sse4_2' /proc/cpuinfo; then
        ARCH_FLAGS="-march=native -msse4.2"
    else
        ARCH_FLAGS=""
    fi
    ARCH_LIBS="-lpthread -lm"

    num_numa_sockets=$(grep '^physical id' /proc/cpuinfo  | sort | uniq | wc -l)
    if ((num_numa_sockets > 1)); then
        ARCH_LIBS="$ARCH_LIBS -lnuma"
    else
        DEFINES="$DEFINES -DDISABLE_NUMA"
    fi
fi

CFLAGS="$ARCH_FLAGS $DEFINES -Wall"

CXXFLAGS="-std=c++20"
LIBS="$ARCH_LIBS"

TARGET=./main

srcs=$(find ./src | egrep '\.cpp$' | tr -s $'\n' ' ')

includes=$(find ./src | egrep '\.(h|hpp)$' | while read HEADER_PATH; do
    HEADER_DIR=$(dirname "$HEADER_PATH")
    echo "-I$HEADER_DIR"
done | sort | uniq | awk '{printf("%s ", $0);}')

rm -f $TARGET
echo "$CC -o $TARGET $srcs $CXXFLAGS $CFLAGS $LIBS $includes $build_flags"
$CC -o $TARGET $srcs $CXXFLAGS $CFLAGS $LIBS $includes $build_flags

if [ -e $TARGET ]; then
    echo -e $'\x1b[32mDone\x1b[0m'
else
    echo -e $'\x1b[31mFailed\x1b[0m'
fi

