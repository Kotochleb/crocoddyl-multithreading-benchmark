#!/bin/bash -eux

cmake -B "build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_WITH_NTHREADS=20 \
    -DCMAKE_C_FLAGS="-O3 -march=native -mavx -mfma" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mavx -mfma" \
    -Wno-dev
cmake --build "build" -j$(nproc)
