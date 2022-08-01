#!/usr/bin/env bash

RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

stamp=$(date +%s)
train -e 15 -b 48 -w 4 -r ${stamp} 004-resnet50/config.yaml peak_scale=5
predict -b 48 -w 4 004-resnet50/predict.yaml snapshots/${stamp}-train/${stamp}-train-14.pt


stamp=$(date +%s)
train -e 15 -b 48 -w 4 -r ${stamp} 004-resnet50/config.yaml peak_scale=10
predict -b 48 -w 4 004-resnet50/predict.yaml snapshots/${stamp}-train/${stamp}-train-14.pt

