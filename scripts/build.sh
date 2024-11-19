#!/bin/bash -eux

BUILD_PATH=$(dirname $(dirname $(realpath $0)))

cmake -B "$BUILD_PATH/build/$1" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_WITH_NTHREADS=20 \
    -DCMAKE_C_FLAGS="-O3 -march=native" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -DCMAKE_CXX_COMPILER="$1" \
    -Wno-dev \
    -Wall
cmake --build "$BUILD_PATH/build/$1" -j$(nproc)
