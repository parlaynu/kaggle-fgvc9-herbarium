#!/usr/bin/env bash

TRIAL_DIR="$( basename "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

# ---------------------------------------------------------------------------------------------
# baseline

stamp=$(date +%s)
train -e 1 -b 96 -w 6 -r ${stamp} "${TRIAL_DIR}/config.yaml"
predict -b 96 -w 4 "${TRIAL_DIR}/predict.yaml" snapshots/${stamp}-train/${stamp}-train-14.pt

