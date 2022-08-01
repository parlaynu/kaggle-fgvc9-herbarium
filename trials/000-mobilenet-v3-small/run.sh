#!/usr/bin/env bash

RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

stamp=$(date +%s)
train -e 15 -b 128 -w 8 -r ${stamp} 000-mobilenet-v3-small/config.yaml
predict -b 128 -w 4 000-mobilenet-v3-small/predict.yaml snapshots/${stamp}-train/${stamp}-train-14.pt

