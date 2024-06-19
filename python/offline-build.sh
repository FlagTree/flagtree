#!/bin/bash

MAX_JOBS=8 TRITON_OFFLINE_BUILD=ON FLAGTREE_OFFLINE_BUILD_DIR=/home/luyunqi/offline-build-3.2.x-x64/ \
           pip install -e . --no-build-isolation -v