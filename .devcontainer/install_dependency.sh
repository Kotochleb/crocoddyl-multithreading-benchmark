#!/bin/bash -eux

CMAKE_PREFIX_PATH=${1:-$PWD/install}

export CMAKE_PREFIX_PATH

IFS="/" read -r -a split <<< "$1"
ORG="${split[0]}"
PRJ="${split[1]}"
TAG="${split[2]}"
CORES="${split[3]}"

git clone --branch "$TAG" --depth 1 "https://github.com/$ORG/$PRJ"
cmake -B "$PRJ/build" -S "$PRJ" \
    -DBUILD_TESTING=OFF \
    -DBUILD_WITH_COLLISION_SUPPORT=ON  \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DHPP_FCL_HAS_QHULL=ON \
    -DGENERATE_PYTHON_STUBS=OFF \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_WITH_MULTITHREADS=ON \
    -DBUILD_WITH_NTHREADS=20 \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -Wno-dev
cmake --build "$PRJ/build" -j $CORES
sudo cmake --build "$PRJ/build" -t install
