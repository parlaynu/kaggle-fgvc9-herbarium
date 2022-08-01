#!/usr/bin/env bash

RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

# predict -b 32 -w 4 003-ensemble/predict-1.yaml snapshots/003-ensemble/003-none.pt
predict -b 32 -w 4 003-ensemble/predict-2.yaml snapshots/003-ensemble/003-mean.pt

